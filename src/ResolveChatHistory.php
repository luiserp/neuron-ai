<?php

declare(strict_types=1);

namespace NeuronAI;

use NeuronAI\Chat\History\AbstractChatHistory;
use NeuronAI\Chat\History\InMemoryChatHistory;
use NeuronAI\Chat\Messages\Message;
use NeuronAI\Observability\Events\MessageSaved;
use NeuronAI\Observability\Events\MessageSaving;

trait ResolveChatHistory
{
    protected AbstractChatHistory $chatHistory;

    /**
     * Called on the agent instance.
     */
    public function withChatHistory(AbstractChatHistory $chatHistory): self
    {
        $this->chatHistory = $chatHistory;
        return $this;
    }

    /**
     * Used extending the Agent.
     */
    protected function chatHistory(): AbstractChatHistory
    {
        return new InMemoryChatHistory();
    }

    public function fillChatHistory(Message|array $messages): void
    {
        $messages = \is_array($messages) ? $messages : [$messages];

        foreach ($messages as $message) {
            $this->notify('message-saving', new MessageSaving($message));
            $this->resolveChatHistory()->addMessage($message);
            $this->notify('message-saved', new MessageSaved($message));
        }
    }

    /**
     * Get the current instance of the chat history.
     */
    public function resolveChatHistory(): AbstractChatHistory
    {
        if (!isset($this->chatHistory)) {
            $this->chatHistory = $this->chatHistory();
        }

        return $this->chatHistory;
    }
}
