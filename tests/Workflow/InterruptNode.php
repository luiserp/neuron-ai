<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Workflow\Node;
use NeuronAI\Workflow\WorkflowState;

class InterruptNode extends Node
{
    public function run(WorkflowState $state): WorkflowState
    {
        $state->set('step', 'interrupt');

        $feedback = $this->interrupt([
            'question' => 'Should we continue?',
            'current_value' => $state->get('value', 0)
        ]);

        $state->set('user_feedback', $feedback);
        return $state;
    }
}
