<?php

declare(strict_types=1);

namespace NeuronAI\Observability\Events;

use NeuronAI\Workflow\Edge;
use NeuronAI\Workflow\NodeInterface;

class WorkflowStart
{
    /**
     * @param NodeInterface[] $nodes
     * @param Edge[] $edges
     */
    public function __construct(public array $nodes, public array $edges)
    {
    }
}
