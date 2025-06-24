<?php

declare(strict_types=1);

namespace NeuronAI\Tools\Toolkits\Riza;

use NeuronAI\Tools\Tool;
use NeuronAI\Tools\Toolkits\AbstractToolkit;

/**
 * @method static make(string $key)
 */
class RizaToolkit extends AbstractToolkit
{
    public function __construct(protected string $key)
    {
    }

    /**
     * @return array<Tool>
     */
    public function provide(): array
    {
        return [
            new RizaCodeInterpreter($this->key),
            new RizaFunctionExecutor($this->key),
        ];
    }
}
