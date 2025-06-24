<?php

declare(strict_types=1);

namespace NeuronAI\Observability\Events;

class AgentError
{
    public function __construct(
        public \Throwable $exception,
        public bool $unhandled = true
    ) {
    }
}
