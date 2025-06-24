<?php

declare(strict_types=1);

namespace NeuronAI\Providers;

use GuzzleHttp\Client;

trait HasGuzzleClient
{
    protected Client $client;

    public function setClient(Client $client): AIProviderInterface
    {
        $this->client = $client;
        return $this;
    }
}
