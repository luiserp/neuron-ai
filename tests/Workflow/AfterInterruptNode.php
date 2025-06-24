<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Workflow\Node;
use NeuronAI\Workflow\WorkflowState;

class AfterInterruptNode extends Node
{
    public function run(WorkflowState $state): WorkflowState
    {
        $state->set('step', 'after_interrupt');
        $state->set('final_value', $state->get('value') + 10);
        return $state;
    }
}
