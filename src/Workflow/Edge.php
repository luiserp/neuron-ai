<?php

declare(strict_types=1);

namespace NeuronAI\Workflow;

class Edge
{
    public function __construct(
        protected string $from,
        protected string $to,
        protected ?\Closure $condition = null
    ) {
    }

    public function getFrom(): string
    {
        return $this->from;
    }

    public function getTo(): string
    {
        return $this->to;
    }

    public function hasCondition(): bool
    {
        return $this->condition !== null;
    }

    public function shouldExecute(WorkflowState $state): bool
    {
        return $this->hasCondition() ? ($this->condition)($state) : true;
    }
}
