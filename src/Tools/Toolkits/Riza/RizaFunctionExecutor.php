<?php

declare(strict_types=1);

namespace NeuronAI\Tools\Toolkits\Riza;

use GuzzleHttp\Client;
use GuzzleHttp\RequestOptions;
use NeuronAI\Tools\PropertyType;
use NeuronAI\Tools\Tool;
use NeuronAI\Tools\ToolProperty;

/**
 * @method static static make(string $pdo, string $language = 'JavaScript')
 */
class RizaFunctionExecutor extends Tool
{
    protected Client $client;

    protected string $url = 'https://api.riza.io/v1/';

    public function __construct(
        protected string $key,
        protected string $language = 'JavaScript', // Python, JavaScript, and TypeScript (no PHP unfortunately)
    ) {
        parent::__construct(
            "execute_{$language}_function",
            "Execute {$language} function and get the result."
        );

    }

    protected function properties(): array
    {
        return [
            new ToolProperty(
                'code',
                PropertyType::STRING,
                'The function code to execute.',
                true,
            ),
            new ToolProperty(
                'input',
                PropertyType::ARRAY,
                'The input arguments to execute the function.',
                false,
            ),
            new ToolProperty(
                'env',
                PropertyType::ARRAY,
                "Set of key-value pairs to add to the script's execution environment.",
                false,
            )
        ];
    }

    protected function getClient(): Client
    {
        return $this->client ?? $this->client = new Client([
            'base_uri' => trim($this->url, '/').'/',
            'headers' => [
                'Authorization' => 'Bearer '.$this->key,
                'Content-Type' => 'application/json; charset=utf-8',
                'Accept' => 'application/json',
            ]
        ]);
    }

    public function __invoke(
        string $code,
        array $input = [],
        array $env = [],
    ) {
        $result = $this->getClient()->post('execute-function', [
            RequestOptions::JSON => [
                'language' => $this->language,
                'code' => $code,
                'input' => $input,
                'env' => $env,
            ]
        ])->getBody()->getContents();

        return \json_decode($result, true);
    }
}
