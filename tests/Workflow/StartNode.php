<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Workflow\Node;
use NeuronAI\Workflow\WorkflowState;

class StartNode extends Node
{
    public function run(WorkflowState $state): WorkflowState
    {
        $state->set('step', 'start');
        return $state;
    }
}
