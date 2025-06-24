<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Exceptions\WorkflowException;
use NeuronAI\Workflow\Edge;
use NeuronAI\Workflow\Persistence\InMemoryPersistence;
use NeuronAI\Workflow\Workflow;
use NeuronAI\Workflow\WorkflowContext;
use NeuronAI\Workflow\WorkflowInterrupt;
use NeuronAI\Workflow\WorkflowState;
use PHPUnit\Framework\TestCase;

class WorkflowHumanInTheLoopTest extends TestCase
{
    public function test_basic_interrupt_and_resume(): void
    {
        $workflow = new Workflow();
        $workflow->addNodes([
                new BeforeInterruptNode(),
                new InterruptNode(),
                new AfterInterruptNode(),
            ])
            ->addEdges([
                new Edge(BeforeInterruptNode::class, InterruptNode::class),
                new Edge(InterruptNode::class, AfterInterruptNode::class),
            ])
            ->setStart(BeforeInterruptNode::class)
            ->setEnd(AfterInterruptNode::class);

        try {
            $workflow->run();
            $this->fail('Expected WorkflowInterrupt exception was not thrown');
        } catch (WorkflowInterrupt $interrupt) {
            $this->assertEquals(InterruptNode::class, $interrupt->getCurrentNode());
            $this->assertEquals([
                'question' => 'Should we continue?',
                'current_value' => 42
            ], $interrupt->getData());

            $state = $interrupt->getState();
            $this->assertEquals('interrupt', $state->get('step'));
            $this->assertEquals(42, $state->get('value'));
        }

        $result = $workflow->resume(['approved' => true]);

        $this->assertEquals('after_interrupt', $result->get('step'));
        $this->assertEquals(['approved' => true], $result->get('user_feedback'));
        $this->assertEquals(52, $result->get('final_value'));
    }

    public function test_workflow_without_interrupt(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new BeforeInterruptNode())
            ->addNode(new AfterInterruptNode())
            ->addEdge(new Edge(BeforeInterruptNode::class, AfterInterruptNode::class))
            ->setStart(BeforeInterruptNode::class)
            ->setEnd(AfterInterruptNode::class);

        $result = $workflow->run();

        $this->assertEquals('after_interrupt', $result->get('step'));
        $this->assertEquals(42, $result->get('value'));
        $this->assertEquals(52, $result->get('final_value'));
    }

    public function test_multiple_interrupts_in_same_workflow(): void
    {
        $workflow = new Workflow();
        $workflow->addNodes([
                new MultipleInterruptNode(),
                new AfterInterruptNode(),
            ])
            ->addEdges([
                // Stay in the loop if the interrupt_counter is lower than 2
                new Edge(
                    MultipleInterruptNode::class,
                    MultipleInterruptNode::class,
                    fn (WorkflowState $state) => $state->get('interrupt_counter', 0) < 2
                ),
                // Move forward when the interrupt_counter is >= 2
                new Edge(
                    MultipleInterruptNode::class,
                    AfterInterruptNode::class,
                    fn (WorkflowState $state) => $state->get('interrupt_counter', 0) >= 2
                )
            ])
            ->setStart(MultipleInterruptNode::class)
            ->setEnd(AfterInterruptNode::class);

        try {
            $workflow->run();
            $this->fail('Expected first WorkflowInterrupt exception was not thrown');
        } catch (WorkflowInterrupt $interrupt) {
            $this->assertEquals([
                'question' => 'Interrupt #0',
                'counter' => 0
            ], $interrupt->getData());
        }

        try {
            $workflow->resume('first_response');
            $this->fail('Expected second WorkflowInterrupt exception was not thrown');
        } catch (WorkflowInterrupt $interrupt) {
            $this->assertEquals([
                'question' => 'Interrupt #1',
                'counter' => 1
            ], $interrupt->getData());
        }

        $finalResult = $workflow->resume('second_response');

        $this->assertEquals('after_interrupt', $finalResult->get('step'));
        $this->assertEquals('first_response', $finalResult->get('feedback_1'));
        $this->assertEquals('second_response', $finalResult->get('feedback_2'));
        $this->assertEquals(2, $finalResult->get('interrupt_counter'));
    }

    public function test_conditional_interrupt(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new ConditionalInterruptNode())
            ->addNode(new AfterInterruptNode())
            ->addEdge(new Edge(ConditionalInterruptNode::class, AfterInterruptNode::class))
            ->setStart(ConditionalInterruptNode::class)
            ->setEnd(AfterInterruptNode::class);

        $lowValueState = new WorkflowState();
        $lowValueState->set('value', 30);

        $result = $workflow->run($lowValueState);
        $this->assertEquals('after_interrupt', $result->get('step'));
        $this->assertFalse($result->has('high_value_feedback'));

        $workflow2 = new Workflow(new InMemoryPersistence(), 'workflow_2');
        $workflow2->addNode(new ConditionalInterruptNode())
            ->addNode(new AfterInterruptNode())
            ->addEdge(new Edge(ConditionalInterruptNode::class, AfterInterruptNode::class))
            ->setStart(ConditionalInterruptNode::class)
            ->setEnd(AfterInterruptNode::class);

        $highValueState = new WorkflowState();
        $highValueState->set('value', 80);

        try {
            $workflow2->run($highValueState);
            $this->fail('Expected WorkflowInterrupt exception was not thrown');
        } catch (WorkflowInterrupt $interrupt) {
            $this->assertEquals([
                'question' => 'Value is high, should we proceed?',
                'value' => 80
            ], $interrupt->getData());
        }

        $result = $workflow2->resume('proceed');
        $this->assertEquals('proceed', $result->get('high_value_feedback'));
    }

    public function test_resume_without_saved_workflow(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new BeforeInterruptNode())
            ->addNode(new AfterInterruptNode())
            ->addEdge(new Edge(BeforeInterruptNode::class, AfterInterruptNode::class))
            ->setStart(BeforeInterruptNode::class)
            ->setEnd(AfterInterruptNode::class);

        $this->expectException(WorkflowException::class);
        $this->expectExceptionMessage('No saved workflow found for ID:');

        $workflow->resume(['some' => 'feedback']);
    }

    public function test_workflow_state_persistence(): void
    {
        $persistence = new InMemoryPersistence();
        $workflow = new Workflow($persistence, 'test_workflow');

        $workflow->addNode(new BeforeInterruptNode())
            ->addNode(new InterruptNode())
            ->addNode(new AfterInterruptNode())
            ->addEdge(new Edge(BeforeInterruptNode::class, InterruptNode::class))
            ->addEdge(new Edge(InterruptNode::class, AfterInterruptNode::class))
            ->setStart(BeforeInterruptNode::class)
            ->setEnd(AfterInterruptNode::class);

        try {
            $workflow->run();
        } catch (WorkflowInterrupt) {
            // Verify interrupt was saved
            $savedInterrupt = $persistence->load('test_workflow');
            $this->assertEquals(InterruptNode::class, $savedInterrupt->getCurrentNode());
        }

        $workflow->resume(['status' => 'approved']);
    }

    public function test_workflow_interrupt_exception(): void
    {
        $state = new WorkflowState();
        $state->set('test', 'value');

        $interrupt = new WorkflowInterrupt(
            ['question' => 'Test question'],
            'TestNode',
            $state
        );

        $this->assertEquals(['question' => 'Test question'], $interrupt->getData());
        $this->assertEquals('TestNode', $interrupt->getCurrentNode());
        $this->assertSame($state, $interrupt->getState());
        $this->assertEquals('Workflow interrupted for human input', $interrupt->getMessage());
    }

    public function test_workflow_context_behavior(): void
    {
        $persistence = new InMemoryPersistence();
        $context = new WorkflowContext('test_id', 'TestNode', $persistence, new WorkflowState());

        try {
            $context->interrupt(['data' => 'test']);
            $this->fail('Expected WorkflowInterrupt exception was not thrown');
        } catch (WorkflowInterrupt $interrupt) {
            $this->assertEquals('TestNode', $interrupt->getCurrentNode());
            $this->assertEquals(['data' => 'test'], $interrupt->getData());
        }

        $context->setResuming(true, ['TestNode' => 'feedback']);
        $result = $context->interrupt(['should' => 'not_matter']);
        $this->assertEquals('feedback', $result);
    }

    public function test_node_without_context(): void
    {
        $node = new InterruptNode();
        $state = new WorkflowState();

        $this->expectException(WorkflowException::class);
        $this->expectExceptionMessage('WorkflowContext not set on node');

        $node->run($state);
    }

    public function test_workflow_id_generation(): void
    {
        $workflow1 = new Workflow();
        $workflow2 = new Workflow();

        $this->assertNotEquals($workflow1->getWorkflowId(), $workflow2->getWorkflowId());
        $this->assertStringStartsWith('neuron_workflow_', $workflow1->getWorkflowId());
    }

    public function testCustomWorkflowId(): void
    {
        $workflow = new Workflow(new InMemoryPersistence(), 'my_custom_id');
        $this->assertEquals('my_custom_id', $workflow->getWorkflowId());
    }
}
