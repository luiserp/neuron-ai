<?php

declare(strict_types=1);

namespace NeuronAI\Providers\OpenAI;

use NeuronAI\Chat\Messages\Message;

trait HandleStructured
{
    public function structured(
        array $messages,
        string $class,
        array $response_format
    ): Message {
        $tk = explode('\\', $class);

        $this->parameters = \array_merge($this->parameters, [
            'response_format' => [
                'type' => 'json_schema',
                'json_schema' => [
                    "name" => end($tk),
                    "strict" => false,
                    "schema" => $response_format,
                ],
            ]
        ]);

        return $this->chat($messages);
    }
}
