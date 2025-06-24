<?php

declare(strict_types=1);

namespace NeuronAI\Chat\History;

use NeuronAI\Chat\Messages\Message;

class InMemoryChatHistory extends AbstractChatHistory
{
    public function __construct(int $contextWindow = 50000)
    {
        parent::__construct($contextWindow);
    }

    protected function storeMessage(Message $message): ChatHistoryInterface
    {
        return $this;
    }

    public function removeOldestMessage(): ChatHistoryInterface
    {
        return $this;
    }

    protected function clear(): ChatHistoryInterface
    {
        return $this;
    }
}
