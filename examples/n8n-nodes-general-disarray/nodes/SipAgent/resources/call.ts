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
		if (callbackUrl) {
			body.callback_url = callbackUrl;
		}
		if (choice) {
			body.choice = choice;
		}

		return sipAgentApiRequest.call(ctx, 'POST', '/call', body);
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
