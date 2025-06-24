<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Workflow\Edge;
use NeuronAI\Workflow\Persistence\FilePersistence;
use NeuronAI\Workflow\Persistence\PersistenceInterface;
use NeuronAI\Workflow\Workflow;
use NeuronAI\Workflow\WorkflowInterrupt;
use NeuronAI\Workflow\WorkflowState;
use PHPUnit\Framework\TestCase;

class WorkflowPersistenceTest extends TestCase
{
    public function test_file_persistence_save(): void
    {
        $persistence = new FilePersistence(__DIR__);
        $this->assertInstanceOf(PersistenceInterface::class, $persistence);

        $interrupt = new WorkflowInterrupt(
            ['question' => 'test'],
            'test_node',
            new WorkflowState()
        );
        $persistence->save('id', $interrupt);
        $this->assertFileExists(__DIR__.DIRECTORY_SEPARATOR.'neuron_workflow_id.store');

        $persistence->delete('id');
        $this->assertFileDoesNotExist(__DIR__.DIRECTORY_SEPARATOR.'neuron_workflow_id.store');
    }

    public function test_file_persistence_load(): void
    {
        $persistence = new FilePersistence(__DIR__);

        $interrupt = new WorkflowInterrupt(
            ['question' => 'test'],
            'test_node',
            new WorkflowState()
        );
        $persistence->save('id', $interrupt);

        $persistence = new FilePersistence(__DIR__);
        $interrupt2 = $persistence->load('id');
        $this->assertEquals($interrupt, $interrupt2);
        $persistence->delete('id');
    }

    public function test_workflow_state_persistence(): void
    {
        $persistence = new FilePersistence(__DIR__);
        $workflow = new Workflow($persistence, 'test_workflow');

        $workflow->addNode(new BeforeInterruptNode())
            ->addNode(new InterruptNode())
            ->addNode(new AfterInterruptNode())
            ->addEdge(new Edge(BeforeInterruptNode::class, InterruptNode::class))
            ->addEdge(new Edge(InterruptNode::class, AfterInterruptNode::class))
            ->setStart(BeforeInterruptNode::class)
            ->setEnd(AfterInterruptNode::class);

        try {
            $workflow->run(new WorkflowState(['value' => 8]));
        } catch (WorkflowInterrupt) {
            // Verify interrupt was saved
            $savedInterrupt = $persistence->load('test_workflow');
            $this->assertEquals(InterruptNode::class, $savedInterrupt->getCurrentNode());
        }

        $result = $workflow->resume(['status' => 'approved']);
        $this->assertEquals($result->get('final_value'), 28);
    }
}
