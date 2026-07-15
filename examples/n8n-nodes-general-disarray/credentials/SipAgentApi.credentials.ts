import type {
	IAuthenticateGeneric,
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

	// Applied to the credential test below. The node's own requests build these
	// headers by hand in GenericFunctions.ts (it needs the same logic for the raw
	// binary /play upload), so the two must stay in step. An unset token yields
	// empty headers, which the agent treats as absent — the supported tokenless mode.
	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: {
			headers: {
				'X-API-Key':
					'={{ $credentials.apiToken && $credentials.headerStyle !== "bearer" ? $credentials.apiToken : "" }}',
				Authorization:
					'={{ $credentials.apiToken && $credentials.headerStyle === "bearer" ? "Bearer " + $credentials.apiToken : "" }}',
			},
		},
	};

	// GET /schedule is token-protected but read-only, so it validates the base URL
	// *and* the token in one shot: 401 on a missing/wrong token, 200 on a correct
	// one — and 200 on a tokenless agent, which is a supported config. (/health is
	// unauthenticated, so testing against it goes green even with a blank token.)
	test: ICredentialTestRequest = {
		request: {
			baseURL: '={{$credentials.baseUrl}}',
			url: '/schedule',
		},
	};
}
