import type { IDataObject, IExecuteFunctions, INodeProperties } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

import { sipAgentApiRequest } from '../GenericFunctions';

export const virtualNumberProperties: INodeProperties[] = [
	{
		displayName: 'Operation',
		name: 'operation',
		type: 'options',
		noDataExpression: true,
		displayOptions: { show: { resource: ['virtualNumber'] } },
		options: [
			{
				name: 'Create',
				value: 'create',
				description:
					'Create an ephemeral inbound extension the agent listens for; single-use, expires on TTL',
				action: 'Create a virtual number',
			},
			{
				name: 'Get Many',
				value: 'getMany',
				description: 'List active virtual numbers',
				action: 'Get many virtual numbers',
			},
			{
				name: 'Get',
				value: 'get',
				description: 'Get a virtual number by ID (404 once used or expired)',
				action: 'Get a virtual number',
			},
			{
				name: 'Delete',
				value: 'delete',
				description: 'Remove a virtual number before it is used',
				action: 'Delete a virtual number',
			},
		],
		default: 'create',
	},

	// ----------------------------------
	//         virtualNumber:create
	// ----------------------------------
	{
		displayName: 'Purpose',
		name: 'purpose',
		type: 'string',
		required: true,
		default: '',
		description:
			'What this number is for — injected into the agent\'s system prompt for the call that arrives on it',
		displayOptions: { show: { resource: ['virtualNumber'], operation: ['create'] } },
	},
	{
		displayName: 'Additional Fields',
		name: 'additionalFields',
		type: 'collection',
		placeholder: 'Add Field',
		default: {},
		displayOptions: { show: { resource: ['virtualNumber'], operation: ['create'] } },
		options: [
			{
				displayName: 'Number',
				name: 'number',
				type: 'string',
				default: '',
				description:
					'Explicit extension (digits/*/#); leave empty to auto-allocate from the agent\'s configured range',
			},
			{
				displayName: 'TTL (Seconds)',
				name: 'ttlS',
				type: 'number',
				typeOptions: { minValue: 1 },
				default: 900,
				description: 'Seconds until the unused number expires',
			},
			{
				displayName: 'Greeting',
				name: 'greeting',
				type: 'string',
				default: '',
				description: 'Custom greeting spoken instead of the default one',
			},
			{
				displayName: 'Callback URL',
				name: 'callbackUrl',
				type: 'string',
				default: '',
				description:
					'URL the agent POSTs the call outcome (status completed/expired, transcript) to',
			},
			{
				displayName: 'Include Transcript',
				name: 'includeTranscript',
				type: 'boolean',
				default: true,
				description: 'Whether the completion webhook includes the call transcript',
			},
			{
				displayName: 'Persistent (Trigger Number)',
				name: 'persistent',
				type: 'boolean',
				default: false,
				description:
					'Whether the number never expires and survives its calls, so every call to it fires the callback until it is deleted (TTL ignored). For a number tied to a workflow\'s lifetime prefer the SIP Agent Trigger node in Trigger Number mode.',
			},
			{
				displayName: 'Events',
				name: 'events',
				type: 'multiOptions',
				options: [
					{ name: 'Call Answered', value: 'answered' },
					{ name: 'First Utterance', value: 'first_speech' },
					{ name: 'Every Utterance', value: 'speech' },
					{ name: 'Call Completed', value: 'completed' },
				],
				default: ['completed'],
				description:
					'Which call-time webhooks the agent POSTs to the Callback URL (anything beyond Call Completed requires one)',
			},
		],
	},

	// ----------------------------------
	//         virtualNumber:get / delete
	// ----------------------------------
	{
		displayName: 'Virtual Number ID',
		name: 'virtualNumberId',
		type: 'string',
		required: true,
		default: '',
		description: 'ID returned when the virtual number was created',
		displayOptions: { show: { resource: ['virtualNumber'], operation: ['get', 'delete'] } },
	},
];

export async function executeVirtualNumber(
	ctx: IExecuteFunctions,
	i: number,
	operation: string,
): Promise<IDataObject | IDataObject[]> {
	if (operation === 'create') {
		const purpose = ctx.getNodeParameter('purpose', i) as string;
		if (!purpose) {
			throw new NodeOperationError(ctx.getNode(), 'Purpose is required', { itemIndex: i });
		}
		const body: IDataObject = { purpose };

		const additionalFields = ctx.getNodeParameter('additionalFields', i, {}) as IDataObject;
		if (additionalFields.number) {
			body.number = additionalFields.number;
		}
		if (additionalFields.ttlS) {
			body.ttl_s = additionalFields.ttlS;
		}
		if (additionalFields.greeting) {
			body.greeting = additionalFields.greeting;
		}
		if (additionalFields.callbackUrl) {
			body.callback_url = additionalFields.callbackUrl;
		}
		if (additionalFields.includeTranscript === false) {
			body.include_transcript = false;
		}
		if (additionalFields.persistent === true) {
			body.persistent = true;
		}
		const events = additionalFields.events as string[] | undefined;
		if (Array.isArray(events) && events.length > 0) {
			body.events = events;
		}

		return sipAgentApiRequest.call(ctx, 'POST', '/virtual-numbers', body);
	}

	if (operation === 'getMany') {
		const response = await sipAgentApiRequest.call(ctx, 'GET', '/virtual-numbers');
		return response as unknown as IDataObject[];
	}

	if (operation === 'get') {
		const virtualNumberId = ctx.getNodeParameter('virtualNumberId', i) as string;
		return sipAgentApiRequest.call(
			ctx,
			'GET',
			`/virtual-numbers/${encodeURIComponent(virtualNumberId)}`,
		);
	}

	if (operation === 'delete') {
		const virtualNumberId = ctx.getNodeParameter('virtualNumberId', i) as string;
		return sipAgentApiRequest.call(
			ctx,
			'DELETE',
			`/virtual-numbers/${encodeURIComponent(virtualNumberId)}`,
		);
	}

	throw new NodeOperationError(ctx.getNode(), `Unknown operation "${operation}"`, {
		itemIndex: i,
	});
}
