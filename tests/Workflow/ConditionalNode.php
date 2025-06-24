<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Workflow\Node;
use NeuronAI\Workflow\WorkflowState;

class ConditionalNode extends Node
{
    public function run(WorkflowState $state): WorkflowState
    {
        $counter = $state->get('counter', 0);
        $state->set('should_loop', $counter < 3);
        return $state;
    }
}
