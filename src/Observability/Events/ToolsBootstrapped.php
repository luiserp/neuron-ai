<?php

declare(strict_types=1);

namespace NeuronAI\Observability\Events;

class ToolsBootstrapped
{
    public function __construct(public array $tools)
    {
    }
}
