<?php

declare(strict_types=1);

namespace NeuronAI\Tools\Toolkits\Tavily;

use NeuronAI\Tools\Tool;
use NeuronAI\Tools\Toolkits\AbstractToolkit;

/**
 * @method static make(string $key)
 */
class TavilyToolkit extends AbstractToolkit
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
            new TavilyExtractTool($this->key),
            new TavilySearchTool($this->key),
        ];
    }
}
