<?php

declare(strict_types=1);

namespace NeuronAI\Tests\Workflow;

use NeuronAI\Workflow\Node;
use NeuronAI\Workflow\WorkflowState;

class MultipleInterruptNode extends Node
{
    public function run(WorkflowState $state): WorkflowState
    {
        $counter = $state->get('interrupt_counter', 0);

        $feedback = $this->interrupt([
            'question' => "Interrupt #{$counter}",
            'counter' => $counter
        ]);

        $state->set('interrupt_counter', ++$counter);

        $state->set("feedback_{$counter}", $feedback);
        return $state;
    }
}
