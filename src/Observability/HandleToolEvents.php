<?php

declare(strict_types=1);

namespace NeuronAI\Observability;

use NeuronAI\Observability\Events\ToolCalled;
use NeuronAI\Observability\Events\ToolCalling;
use NeuronAI\Observability\Events\ToolsBootstrapped;
use NeuronAI\Tools\ToolInterface;

trait HandleToolEvents
{
    public function toolsBootstrapping(\NeuronAI\AgentInterface $agent, string $event, mixed $data): void
    {
        if (!$this->inspector->canAddSegments()) {
            return;
        }

        $this->segments[$agent::class.'_tools_bootstrap'] = $this->inspector
            ->startSegment(
                self::SEGMENT_TYPE.'-tools-bootstrap',
                "toolsBootstrap()"
            )
            ->setColor(self::SEGMENT_COLOR);
    }

    public function toolsBootstrapped(\NeuronAI\AgentInterface $agent, string $event, ToolsBootstrapped $data): void
    {
        if (\array_key_exists($agent::class.'_tools_bootstrap', $this->segments) && !empty($data->tools)) {
            $segment = $this->segments[$agent::class.'_tools_bootstrap']->end();
            $segment->addContext('Tools', \array_map(fn (ToolInterface $tool) => $tool->getName(), $data->tools));
        }
    }

    public function toolCalling(\NeuronAI\AgentInterface $agent, string $event, ToolCalling $data): void
    {
        if (!$this->inspector->canAddSegments()) {
            return;
        }

        $this->segments[$data->tool->getName()] = $this->inspector
            ->startSegment(
                self::SEGMENT_TYPE.'-tool-call',
                "toolCall( {$data->tool->getName()} )"
            )
            ->setColor(self::SEGMENT_COLOR);
    }

    public function toolCalled(\NeuronAI\AgentInterface $agent, string $event, ToolCalled $data): void
    {
        if (\array_key_exists($data->tool->getName(), $this->segments)) {
            $this->segments[$data->tool->getName()]
                ->end()
                ->addContext('Tool', $data->tool->jsonSerialize());
        }
    }
}
