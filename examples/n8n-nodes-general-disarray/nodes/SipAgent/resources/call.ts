import type { IDataObject, IExecuteFunctions, INodeProperties } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

import { sipAgentApiRequest } from '../GenericFunctions';
import { assertCallbackWithChoice, buildChoice, choiceFixedCollection } from '../shared';

export const callProperties: INodeProperties[] = [
	{
		displayName: 'Operation',
		name: 'operation',
		type: 'options',
		noDataExpression: true,
		displayOptions: { show: { resource: ['call'] } },
		options: [
			{
				name: 'Make',
				value: 'make',
				description: 'Make an outbound call that speaks a message',
				action: 'Make a call',
			},
			{
				name: 'Verify',
				value: 'verify',
				description:
					'Call the person, prompt them to key in their PIN or one-time code, and return success or fail',
				action: 'Call and verify a caller',
			},
			{
				name: 'Get Status',
				value: 'getStatus',
				description: 'Get the status of a call',
				action: 'Get call status',
			},
			{
				name: 'Get Transcript',
				value: 'getTranscript',
				description: 'Get the conversation transcript of a call',
				action: 'Get call transcript',
			},
		],
		default: 'make',
	},

	// ----------------------------------
	//         call:make
	// ----------------------------------
	{
		displayName: 'Message',
		name: 'message',
		type: 'string',
		required: true,
		default: '',
		description: 'The message to speak when the callee answers',
		displayOptions: { show: { resource: ['call'], operation: ['make'] } },
	},
	{
		displayName: 'Extension',
		name: 'extension',
		type: 'string',
		required: true,
		default: '',
		description: 'SIP extension or phone number to call',
		displayOptions: { show: { resource: ['call'], operation: ['make'] } },
	},
	{
		displayName: 'Callback URL',
		name: 'callbackUrl',
		type: 'string',
		default: '',
		description:
			'URL the agent POSTs the call result to (required when a Choice Prompt is set; point it at a SIP Agent Trigger node)',
		displayOptions: { show: { resource: ['call'], operation: ['make'] } },
	},
	choiceFixedCollection(['call'], ['make']),
	{
		displayName: 'Additional Fields',
		name: 'additionalFields',
		type: 'collection',
		placeholder: 'Add Field',
		default: {},
		displayOptions: { show: { resource: ['call'], operation: ['make'] } },
		options: [
			{
				displayName: 'Reformat for Speech',
				name: 'reformatForSpeech',
				type: 'boolean',
				default: false,
				description:
					'Whether the agent\'s LLM rewrites the message into natural spoken form (dates, numbers, URLs, IDs said aloud) without dropping any information. Adds a moment of latency; falls back to the original text on failure.',
			},
			{
				displayName: 'Ring Timeout (Seconds)',
				name: 'ringTimeout',
				type: 'number',
				typeOptions: { minValue: 1, maxValue: 600 },
				default: 30,
				description: 'How long to let the phone ring before giving up',
			},
			{
				displayName: 'Call ID',
				name: 'callId',
				type: 'string',
				default: '',
				description:
					'Custom call ID (letters, digits, dots, underscores, dashes; max 64 chars). Auto-generated when empty.',
			},
			{
				displayName: 'Caller Name',
				name: 'callerName',
				type: 'string',
				default: '',
				placeholder: 'Weather Alert',
				description:
					'Name shown on the recipient\'s phone instead of the agent\'s extension, e.g. "Weather Alert". Applies to this call only. An internal PBX passes it through to the handset; a PSTN carrier will usually replace it with its own CNAM lookup.',
			},
		],
	},

	// ----------------------------------
	//         call:verify
	// ----------------------------------
	{
		displayName: 'Caller ID',
		name: 'callerId',
		type: 'string',
		default: '',
		placeholder: '1001',
		description:
			'Caller id whose stored PIN/OTP credentials are checked (the user part of the SIP URI). Optional when you supply an inline PIN/TOTP Secret and an Extension — then it defaults to the Extension. Either Caller ID or Extension is required.',
		displayOptions: { show: { resource: ['call'], operation: ['verify'] } },
	},
	{
		displayName: 'Extension',
		name: 'verifyExtension',
		type: 'string',
		default: '',
		placeholder: '1001',
		description: 'SIP extension or number to dial. Defaults to the Caller ID when empty.',
		displayOptions: { show: { resource: ['call'], operation: ['verify'] } },
	},
	{
		displayName: 'Factor',
		name: 'method',
		type: 'options',
		options: [
			{ name: 'Either (Auto)', value: 'auto' },
			{ name: 'PIN Only', value: 'pin' },
			{ name: 'One-Time Code Only', value: 'otp' },
		],
		default: 'auto',
		description: 'Which factor the person must key in',
		displayOptions: { show: { resource: ['call'], operation: ['verify'] } },
	},
	{
		displayName: 'PIN',
		name: 'verifyPin',
		type: 'string',
		typeOptions: { password: true },
		default: '',
		description:
			'Check the entered code against this PIN for this call. Leave empty to use the caller\'s stored/global credentials.',
		displayOptions: { show: { resource: ['call'], operation: ['verify'] } },
	},
	{
		displayName: 'TOTP Secret',
		name: 'verifyTotpSecret',
		type: 'string',
		typeOptions: { password: true },
		default: '',
		description:
			'Check the entered code against this base32 TOTP secret for this call. Leave empty to use the caller\'s stored/global credentials.',
		displayOptions: { show: { resource: ['call'], operation: ['verify'] } },
	},
	{
		displayName: 'TOTP Options',
		name: 'verifyTotpOptions',
		type: 'collection',
		placeholder: 'Add TOTP Option',
		default: {},
		description:
			'TOTP algorithm parameters for the inline TOTP Secret. Each defaults to the server\'s VERIFY_TOTP_* config; override to match a non-standard authenticator.',
		displayOptions: { show: { resource: ['call'], operation: ['verify'] } },
		options: [
			{
				displayName: 'Digits',
				name: 'digits',
				type: 'number',
				typeOptions: { minValue: 4, maxValue: 10 },
				default: 6,
				description: 'Number of digits in the code',
			},
			{
				displayName: 'Period (Seconds)',
				name: 'period',
				type: 'number',
				typeOptions: { minValue: 5, maxValue: 300 },
				default: 30,
				description: 'Seconds per TOTP step',
			},
			{
				displayName: 'Algorithm',
				name: 'algorithm',
				type: 'options',
				options: [
					{ name: 'SHA1', value: 'SHA1' },
					{ name: 'SHA256', value: 'SHA256' },
					{ name: 'SHA512', value: 'SHA512' },
				],
				default: 'SHA1',
				description: 'HMAC hash algorithm',
			},
			{
				displayName: 'Skew Window',
				name: 'window',
				type: 'number',
				typeOptions: { minValue: 0, maxValue: 10 },
				default: 1,
				description: 'Number of steps of clock skew to accept on each side',
			},
		],
	},
	{
		displayName: 'Callback URL',
		name: 'verifyCallbackUrl',
		type: 'string',
		default: '',
		description:
			'Optional URL the agent also POSTs the verify result to (the result is returned synchronously regardless)',
		displayOptions: { show: { resource: ['call'], operation: ['verify'] } },
	},
	{
		displayName: 'Spoken Messages',
		name: 'verifyMessages',
		type: 'collection',
		placeholder: 'Override a Message',
		default: {},
		description: 'Optionally override the default spoken lines (each falls back to the server default when empty)',
		displayOptions: { show: { resource: ['call'], operation: ['verify'] } },
		options: [
			{
				displayName: 'Prompt',
				name: 'prompt',
				type: 'string',
				default: '',
				placeholder: 'Please enter your PIN or one-time code, then press pound.',
				description: 'Asked when requesting the code',
			},
			{
				displayName: 'Retry Phrase',
				name: 'retryPhrase',
				type: 'string',
				default: '',
				placeholder: 'That code wasn\'t right. Please try again.',
				description: 'Spoken after a wrong code when attempts remain',
			},
			{
				displayName: 'Success Phrase',
				name: 'successPhrase',
				type: 'string',
				default: '',
				placeholder: 'Thank you — your identity is verified. Goodbye.',
				description: 'Spoken when verification succeeds',
			},
			{
				displayName: 'Fail Phrase',
				name: 'failPhrase',
				type: 'string',
				default: '',
				placeholder: 'I could not verify your identity. Goodbye.',
				description: 'Spoken when verification fails',
			},
			{
				displayName: 'Ring Timeout (Seconds)',
				name: 'ringTimeout',
				type: 'number',
				typeOptions: { minValue: 1, maxValue: 600 },
				default: 30,
				description: 'How long to let the phone ring before giving up',
			},
		],
	},

	// ----------------------------------
	//         call:getStatus
	// ----------------------------------
	{
		displayName: 'Call ID',
		name: 'callId',
		type: 'string',
		required: true,
		default: '',
		description: 'The ID of the call',
		displayOptions: { show: { resource: ['call'], operation: ['getStatus'] } },
	},
	{
		displayName: 'Error on Not Found',
		name: 'errorOnNotFound',
		type: 'boolean',
		default: false,
		description:
			'Whether to throw an error when the call is unknown. The API returns 200 with status "not_found" instead of an HTTP 404.',
		displayOptions: { show: { resource: ['call'], operation: ['getStatus'] } },
	},

	// ----------------------------------
	//         call:getTranscript
	// ----------------------------------
	{
		displayName: 'Call ID',
		name: 'callId',
		type: 'string',
		required: true,
		default: '',
		description: 'The ID of the call',
		displayOptions: { show: { resource: ['call'], operation: ['getTranscript'] } },
	},
];

export async function executeCall(
	ctx: IExecuteFunctions,
	i: number,
	operation: string,
): Promise<IDataObject | IDataObject[]> {
	if (operation === 'make') {
		const message = ctx.getNodeParameter('message', i) as string;
		const extension = ctx.getNodeParameter('extension', i) as string;
		const callbackUrl = ctx.getNodeParameter('callbackUrl', i, '') as string;
		const additionalFields = ctx.getNodeParameter('additionalFields', i, {}) as IDataObject;

		const choice = buildChoice(ctx, i);
		assertCallbackWithChoice(ctx, i, choice, callbackUrl);

		const body: IDataObject = {
			message,
			extension,
			ring_timeout: (additionalFields.ringTimeout as number) ?? 30,
		};
		if (additionalFields.callId) {
			body.call_id = additionalFields.callId;
		}
		if (additionalFields.callerName) {
			body.caller_name = additionalFields.callerName;
		}
		if (additionalFields.reformatForSpeech) {
			body.reformat_for_speech = true;
		}
		if (callbackUrl) {
			body.callback_url = callbackUrl;
		}
		if (choice) {
			body.choice = choice;
		}

		return sipAgentApiRequest.call(ctx, 'POST', '/call', body);
	}

	if (operation === 'verify') {
		const callerId = ctx.getNodeParameter('callerId', i, '') as string;
		const extension = ctx.getNodeParameter('verifyExtension', i, '') as string;
		if (!callerId && !extension) {
			throw new NodeOperationError(
				ctx.getNode(),
				'Provide a Caller ID or an Extension to dial',
				{ itemIndex: i },
			);
		}
		const method = ctx.getNodeParameter('method', i, 'auto') as string;
		const pin = ctx.getNodeParameter('verifyPin', i, '') as string;
		const totpSecret = ctx.getNodeParameter('verifyTotpSecret', i, '') as string;
		const totp = ctx.getNodeParameter('verifyTotpOptions', i, {}) as IDataObject;
		const callbackUrl = ctx.getNodeParameter('verifyCallbackUrl', i, '') as string;
		const messages = ctx.getNodeParameter('verifyMessages', i, {}) as IDataObject;

		const body: IDataObject = { method };
		if (callerId) body.caller_id = callerId;
		if (extension) body.extension = extension;
		if (pin) body.pin = pin;
		if (totpSecret) body.totp_secret = totpSecret;
		if (totp.digits) body.totp_digits = totp.digits;
		if (totp.period) body.totp_period = totp.period;
		if (totp.algorithm) body.totp_algorithm = totp.algorithm;
		if (totp.window !== undefined && totp.window !== '') body.totp_window = totp.window;
		if (callbackUrl) body.callback_url = callbackUrl;
		if (messages.prompt) body.prompt = messages.prompt;
		if (messages.retryPhrase) body.retry_phrase = messages.retryPhrase;
		if (messages.successPhrase) body.success_phrase = messages.successPhrase;
		if (messages.failPhrase) body.fail_phrase = messages.failPhrase;
		if (messages.ringTimeout) body.ring_timeout = messages.ringTimeout;

		return sipAgentApiRequest.call(ctx, 'POST', '/verify/call', body);
	}

	if (operation === 'getStatus') {
		const callId = ctx.getNodeParameter('callId', i) as string;
		const errorOnNotFound = ctx.getNodeParameter('errorOnNotFound', i, false) as boolean;

		const response = await sipAgentApiRequest.call(
			ctx,
			'GET',
			`/call/${encodeURIComponent(callId)}`,
		);
		if (errorOnNotFound && response.status === 'not_found') {
			throw new NodeOperationError(ctx.getNode(), `Call "${callId}" was not found`, {
				itemIndex: i,
			});
		}
		return response;
	}

	if (operation === 'getTranscript') {
		const callId = ctx.getNodeParameter('callId', i) as string;
		return sipAgentApiRequest.call(
			ctx,
			'GET',
			`/call/${encodeURIComponent(callId)}/transcript`,
		);
	}

	throw new NodeOperationError(ctx.getNode(), `Unknown operation "${operation}"`, {
		itemIndex: i,
	});
}
