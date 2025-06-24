<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Workflow\Node;
use NeuronAI\Workflow\WorkflowState;

class BeforeInterruptNode extends Node
{
    public function run(WorkflowState $state): WorkflowState
    {
        $state->set('step', 'before_interrupt');

        if ($state->has('value')) {
            $state->set('value', $state->get('value') + 10);
        } else {
            $state->set('value', 42);
        }

        return $state;
    }
}
