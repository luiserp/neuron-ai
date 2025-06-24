<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Workflow\Node;
use NeuronAI\Workflow\WorkflowState;

class MiddleNode extends Node
{
    public function run(WorkflowState $state): WorkflowState
    {
        $state->set('step', 'middle');
        $state->set('counter', $state->get('counter', 0) + 1);
        return $state;
    }
}
