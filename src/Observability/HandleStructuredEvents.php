<?php

declare(strict_types=1);

namespace NeuronAI\Observability;

use NeuronAI\Observability\Events\Deserialized;
use NeuronAI\Observability\Events\Deserializing;
use NeuronAI\Observability\Events\Extracted;
use NeuronAI\Observability\Events\Extracting;
use NeuronAI\Observability\Events\Validated;
use NeuronAI\Observability\Events\Validating;

trait HandleStructuredEvents
{
    protected function extracting(\NeuronAI\AgentInterface $agent, string $event, Extracting $data): void
    {
        $id = $this->getMessageId($data->message).'-extract';

        $this->segments[$id] = $this->inspector->startSegment('structured-extract')
            ->setColor(self::SEGMENT_COLOR);
    }

    protected function extracted(\NeuronAI\AgentInterface $agent, string $event, Extracted $data): void
    {
        $id = $this->getMessageId($data->message).'-extract';

        if (\array_key_exists($id, $this->segments)) {
            $segment = $this->segments[$id]->end();
            $segment->addContext(
                'Data',
                [
                    'response' => $data->message->jsonSerialize(),
                    'json' => $data->json,
                ]
            )->addContext(
                'Schema',
                $data->schema
            );
            unset($this->segments[$id]);
        }
    }

    protected function deserializing(\NeuronAI\AgentInterface $agent, string $event, Deserializing $data): void
    {
        $this->segments[$data->class.'-deserialize'] = $this->inspector->startSegment('structured-deserialize', "deserialize( {$data->class} )")
            ->setColor(self::SEGMENT_COLOR);
    }

    protected function deserialized(\NeuronAI\AgentInterface $agent, string $event, Deserialized $data): void
    {
        $id = $data->class.'-deserialize';

        if (\array_key_exists($id, $this->segments)) {
            $this->segments[$id]->end();
        }
    }

    protected function validating(\NeuronAI\AgentInterface $agent, string $event, Validating $data): void
    {
        $this->segments[$data->class.'-validate'] = $this->inspector->startSegment('structured-validate', "validate( {$data->class} )")
        ->setColor(self::SEGMENT_COLOR)->setColor(self::SEGMENT_COLOR);
    }

    protected function validated(\NeuronAI\AgentInterface $agent, string $event, Validated $data): void
    {
        $id = $data->class.'-validate';

        if (\array_key_exists($id, $this->segments)) {
            $segment = $this->segments[$id]->end();
            $segment->addContext('Json', \json_decode($data->json));
            if (!empty($data->violations)) {
                $segment->addContext('Violations', $data->violations);
            }
        }
    }
}
