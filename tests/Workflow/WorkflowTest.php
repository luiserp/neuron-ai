<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Exceptions\WorkflowException;
use NeuronAI\Workflow\Edge;
use NeuronAI\Workflow\Exporter\ExporterInterface;
use NeuronAI\Workflow\Workflow;
use NeuronAI\Workflow\WorkflowState;
use PHPUnit\Framework\TestCase;

class WorkflowTest extends TestCase
{
    public function test_basic_workflow(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new StartNode())
            ->addNode(new FinishNode())
            ->addEdge(new Edge(StartNode::class, FinishNode::class))
            ->setStart(StartNode::class)
            ->setEnd(FinishNode::class);

        $result = $workflow->run();

        $this->assertEquals('end', $result->get('step'));
    }

    public function test_workflow_initial_state(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new StartNode())
            ->addNode(new FinishNode())
            ->addEdge(new Edge(StartNode::class, FinishNode::class))
            ->setStart(StartNode::class)
            ->setEnd(FinishNode::class);

        $initialState = new WorkflowState();
        $initialState->set('initial_value', 'test');

        $result = $workflow->run($initialState);

        $this->assertEquals('test', $result->get('initial_value'));
        $this->assertEquals('end', $result->get('step'));
    }

    public function test_workflow_multiple_nodes(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new StartNode())
            ->addNode(new MiddleNode())
            ->addNode(new FinishNode())
            ->addEdge(new Edge(StartNode::class, MiddleNode::class))
            ->addEdge(new Edge(MiddleNode::class, FinishNode::class))
            ->setStart(StartNode::class)
            ->setEnd(FinishNode::class);

        $result = $workflow->run();

        $this->assertEquals('end', $result->get('step'));
        $this->assertEquals(1, $result->get('counter'));
    }

    public function testWorkflowWithConditionalEdges(): void
    {
        $workflow = new Workflow();
        $workflow->addNodes([
                new StartNode(),
                new MiddleNode(),
                new ConditionalNode(),
                new FinishNode(),
            ])
            ->addEdges([
                new Edge(StartNode::class, MiddleNode::class),
                new Edge(MiddleNode::class, ConditionalNode::class),
                new Edge(
                    ConditionalNode::class,
                    MiddleNode::class,
                    fn (WorkflowState $state) => $state->get('should_loop', false)
                ),
                new Edge(
                    ConditionalNode::class,
                    FinishNode::class,
                    fn (WorkflowState $state) => !$state->get('should_loop', false)
                )
            ])
            ->setStart(StartNode::class)
            ->setEnd(FinishNode::class);

        $result = $workflow->run();

        $this->assertEquals('end', $result->get('step'));
        $this->assertEquals(3, $result->get('counter'));
        $this->assertFalse($result->get('should_loop'));
    }

    public function test_validation_throws_exception_when_start_node_not_set(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new StartNode())
            ->addNode(new FinishNode())
            ->addEdge(new Edge(StartNode::class, FinishNode::class))
            ->setEnd(FinishNode::class);

        $this->expectException(WorkflowException::class);
        $this->expectExceptionMessage('Start node must be defined');

        $workflow->run();
    }

    public function test_validation_throws_exception_when_end_node_not_set(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new StartNode())
            ->addNode(new FinishNode())
            ->addEdge(new Edge(StartNode::class, FinishNode::class))
            ->setStart(StartNode::class);

        $this->expectException(WorkflowException::class);
        $this->expectExceptionMessage('End node must be defined');

        $workflow->run();
    }

    public function test_validation_throws_exception_when_start_node_does_not_exist(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new FinishNode())
            ->addEdge(new Edge(StartNode::class, FinishNode::class))
            ->setStart(StartNode::class)
            ->setEnd(FinishNode::class);

        $this->expectException(WorkflowException::class);
        $this->expectExceptionMessage("Start node ".StartNode::class." does not exist");

        $workflow->run();
    }

    public function test_validation_throws_exception_when_end_node_does_not_exist(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new StartNode())
            ->addEdge(new Edge(StartNode::class, FinishNode::class))
            ->setStart(StartNode::class)
            ->setEnd(FinishNode::class);

        $this->expectException(WorkflowException::class);
        $this->expectExceptionMessage("End node ".FinishNode::class." does not exist");

        $workflow->run();
    }

    public function test_validation_throws_exception_when_edge_from_node_does_not_exist(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new FinishNode())
            ->addEdge(new Edge(StartNode::class, FinishNode::class))
            ->setStart(FinishNode::class)
            ->setEnd(FinishNode::class);

        $this->expectException(WorkflowException::class);
        $this->expectExceptionMessage("Edge from node ".StartNode::class." does not exist");

        $workflow->run();
    }

    public function test_validation_throws_exception_when_edge_to_node_does_not_exist(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new StartNode())
            ->addEdge(new Edge(StartNode::class, FinishNode::class))
            ->setStart(StartNode::class)
            ->setEnd(StartNode::class);

        $this->expectException(WorkflowException::class);
        $this->expectExceptionMessage("Edge to node ".FinishNode::class." does not exist");

        $workflow->run();
    }

    public function test_execution_throws_exception_when_no_valid_edge_found(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new StartNode())
            ->addNode(new FinishNode())
            ->addEdge(new Edge(
                StartNode::class,
                FinishNode::class,
                fn (WorkflowState $state) => false
            ))
            ->setStart(StartNode::class)
            ->setEnd(FinishNode::class);

        $this->expectException(WorkflowException::class);
        $this->expectExceptionMessage("No valid edge found from node ".StartNode::class);

        $workflow->run();
    }

    public function testMermaidExport(): void
    {
        $workflow = new Workflow();
        $workflow->addNode(new StartNode())
            ->addNode(new MiddleNode())
            ->addNode(new FinishNode())
            ->addEdge(new Edge(StartNode::class, MiddleNode::class))
            ->addEdge(new Edge(MiddleNode::class, FinishNode::class))
            ->setStart(StartNode::class)
            ->setEnd(FinishNode::class);

        $mermaid = $workflow->export();

        $this->assertStringContainsString('graph TD', $mermaid);
        $this->assertStringContainsString('StartNode --> MiddleNode', $mermaid);
        $this->assertStringContainsString('MiddleNode --> FinishNode', $mermaid);
    }

    public function test_custom_exporter(): void
    {
        $customExporter = new class () implements ExporterInterface {
            public function export(Workflow $graph): string
            {
                return 'custom export';
            }
        };

        $workflow = new Workflow();
        $workflow->setExporter($customExporter);
        $result = $workflow->export();

        $this->assertEquals('custom export', $result);
    }

    public function test_workflow_state_data_management(): void
    {
        $state = new WorkflowState();

        $state->set('key1', 'value1');
        $state->set('key2', 42);

        $this->assertEquals('value1', $state->get('key1'));
        $this->assertEquals(42, $state->get('key2'));
        $this->assertEquals('default', $state->get('nonexistent', 'default'));
        $this->assertTrue($state->has('key1'));
        $this->assertFalse($state->has('nonexistent'));

        $all = $state->all();
        $this->assertEquals(['key1' => 'value1', 'key2' => 42], $all);
    }

    public function test_edge_condition_evaluation(): void
    {
        $state = new WorkflowState();
        $state->set('test_value', true);

        $edge = new Edge(
            StartNode::class,
            FinishNode::class,
            fn (WorkflowState $s) => $s->get('test_value', false)
        );

        $this->assertTrue($edge->shouldExecute($state));

        $state->set('test_value', false);
        $this->assertFalse($edge->shouldExecute($state));
    }

    public function test_edge_without_condition(): void
    {
        $state = new WorkflowState();
        $edge = new Edge(StartNode::class, FinishNode::class);

        $this->assertTrue($edge->shouldExecute($state));
    }

    public function test_get_edges_and_nodes(): void
    {
        $workflow = new Workflow();
        $startNode = new StartNode();
        $endNode = new FinishNode();
        $edge = new Edge(StartNode::class, FinishNode::class);

        $workflow->addNode($startNode)
            ->addNode($endNode)
            ->addEdge($edge);

        $edges = $workflow->getEdges();
        $nodes = $workflow->getNodes();

        $this->assertCount(1, $edges);
        $this->assertSame($edge, $edges[0]);

        $this->assertCount(2, $nodes);
        $this->assertSame($startNode, $nodes[StartNode::class]);
        $this->assertSame($endNode, $nodes[FinishNode::class]);
    }
}
