<?php

declare(strict_types=1);

namespace NeuronAI;

use NeuronAI\Providers\AIProviderInterface;

trait ResolveProvider
{
    /**
     * The AI provider instance.
     *
     * @var AIProviderInterface
     */
    protected AIProviderInterface $provider;

    /**
     * @deprecated
     */
    public function withProvider(AIProviderInterface $provider): AgentInterface
    {
        $this->provider = $provider;
        return $this;
    }

    public function setAiProvider(AIProviderInterface $provider): AgentInterface
    {
        $this->provider = $provider;
        return $this;
    }

    protected function provider(): AIProviderInterface
    {
        return $this->provider;
    }

    /**
     * Get the current instance of the chat history.
     *
     * @return AIProviderInterface
     */
    public function resolveProvider(): AIProviderInterface
    {
        if (!isset($this->provider)) {
            $this->provider = $this->provider();
        }

        return $this->provider;
    }
}
