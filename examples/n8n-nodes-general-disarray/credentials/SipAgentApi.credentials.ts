import type {
	ICredentialTestRequest,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class SipAgentApi implements ICredentialType {
	name = 'sipAgentApi';

	displayName = 'SIP Agent API';

	documentationUrl = 'https://github.com/cha0s-corp/general-disarray';

	properties: INodeProperties[] = [
		{
			displayName: 'Base URL',
			name: 'baseUrl',
			type: 'string',
			default: 'http://sip-agent:8080',
			description:
				'Base URL of the SIP agent REST API. From inside the docker network use http://sip-agent:8080.',
		},
		{
			displayName: 'API Token',
			name: 'apiToken',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			description:
				'Value of API_AUTH_TOKEN on the agent. Leave empty if the agent runs without authentication.',
		},
		{
			displayName: 'Auth Header Style',
			name: 'headerStyle',
			type: 'options',
			options: [
				{ name: 'X-API-Key', value: 'xApiKey' },
				{ name: 'Authorization: Bearer', value: 'bearer' },
			],
			default: 'xApiKey',
			description: 'How the API token is sent. The agent accepts both.',
		},
		{
			displayName: 'Webhook Signing Secret',
			name: 'signingSecret',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			description:
				'Value of WEBHOOK_SIGNING_SECRET on the agent. Used only by the SIP Agent Trigger node to verify HMAC signatures on incoming callbacks. Leave empty if signing is disabled.',
		},
	];

	// /health is unauthenticated, so this validates connectivity and base URL only,
	// not token correctness (all authenticated endpoints are mutating POSTs).
	test: ICredentialTestRequest = {
		request: {
			baseURL: '={{$credentials.baseUrl}}',
			url: '/health',
		},
	};
}
