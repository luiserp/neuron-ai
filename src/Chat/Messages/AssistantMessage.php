<?php

declare(strict_types=1);

namespace NeuronAI\Chat\Messages;

use NeuronAI\Chat\Enums\MessageRole;

class AssistantMessage extends Message
{
    public function __construct(array|string|int|float|null $content)
    {
        parent::__construct(MessageRole::ASSISTANT, $content);
    }
}
