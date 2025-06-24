<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Workflow\Node;
use NeuronAI\Workflow\WorkflowState;

class ConditionalInterruptNode extends Node
{
    public function run(WorkflowState $state): WorkflowState
    {
        $value = $state->get('value', 0);
        $state->set('step', 'conditional');

        if ($value > 50) {
            $feedback = $this->interrupt([
                'question' => 'Value is high, should we proceed?',
                'value' => $value
            ]);
            $state->set('high_value_feedback', $feedback);
        }

        return $state;
    }
}
