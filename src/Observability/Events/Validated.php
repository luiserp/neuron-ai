<?php

declare(strict_types=1);

namespace NeuronAI\Observability\Events;

class Validated
{
    /**
     * @param string $class
     * @param string $json
     * @param array<string> $violations
     */
    public function __construct(
        public string $class,
        public string $json,
        public array $violations = []
    ) {
    }
}
