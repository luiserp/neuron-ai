<?php

declare(strict_types=1);

namespace NeuronAI;

trait StaticConstructor
{
    /**
     * Static constructor.
     *
     * @param ...$arguments
     * @return static
     */
    public static function make(...$arguments): static
    {
        /** @phpstan-ignore new.static */
        return new static(...$arguments);
    }
}
