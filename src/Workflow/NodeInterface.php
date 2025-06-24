<?php

declare(strict_types=1);

namespace NeuronAI\Workflow;

interface NodeInterface
{
    public function run(WorkflowState $state): WorkflowState;
    public function setContext(WorkflowContext $context): void;
}
