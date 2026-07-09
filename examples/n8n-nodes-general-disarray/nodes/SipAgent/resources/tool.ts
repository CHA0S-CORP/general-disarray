import type { IDataObject, IExecuteFunctions, INodeProperties } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

import { sipAgentApiRequest } from '../GenericFunctions';
import {
	assertCallbackWithChoice,
	buildChoice,
	choiceFixedCollection,
	parseJsonParameter,
} from '../shared';

export const toolProperties: INodeProperties[] = [
	{
		displayName: 'Operation',
		name: 'operation',
		type: 'options',
		noDataExpression: true,
		displayOptions: { show: { resource: ['tool'] } },
		options: [
			{
				name: 'Get Many',
				value: 'getMany',
				description: 'List all registered tools',
				action: 'Get many tools',
			},
			{
				name: 'Get',
				value: 'get',
				description: 'Get info about a single tool',
				action: 'Get a tool',
			},
			{
				name: 'Execute',
				value: 'execute',
				description: 'Execute a tool directly',
				action: 'Execute a tool',
			},
			{
				name: 'Execute and Call',
				value: 'executeAndCall',
				description: 'Execute a tool, then place an outbound call speaking the result',
				action: 'Execute a tool and call',
			},
		],
		default: 'getMany',
	},
	{
		displayName: 'Tool Name',
		name: 'toolName',
		type: 'string',
		required: true,
		default: '',
		placeholder: 'WEATHER',
		description:
			'Name of the tool. Tool names are uppercase, e.g. WEATHER, TIMER (the server uppercases anyway).',
		displayOptions: {
			show: { resource: ['tool'], operation: ['get', 'execute', 'executeAndCall'] },
		},
	},
	{
		displayName: 'Tool Parameters',
		name: 'toolParams',
		type: 'json',
		default: '{}',
		description: 'Parameters to pass to the tool, as a JSON object',
		displayOptions: {
			show: { resource: ['tool'], operation: ['execute', 'executeAndCall'] },
		},
	},
	{
		displayName: 'Speak Result',
		name: 'speakResult',
		type: 'boolean',
		default: false,
		description: 'Whether to speak the tool result into the active call',
		displayOptions: { show: { resource: ['tool'], operation: ['execute'] } },
	},
	{
		displayName: 'Call ID',
		name: 'callId',
		type: 'string',
		default: '',
		description: 'Optional call ID to associate the execution with',
		displayOptions: { show: { resource: ['tool'], operation: ['execute'] } },
	},
	{
		displayName: 'Extension',
		name: 'extension',
		type: 'string',
		required: true,
		default: '',
		placeholder: '1001',
		description: 'SIP extension or number to call with the tool result',
		displayOptions: { show: { resource: ['tool'], operation: ['executeAndCall'] } },
	},
	{
		displayName: 'Callback URL',
		name: 'callbackUrl',
		type: 'string',
		default: '',
		description:
			'URL the agent POSTs the call result to. Required when a Choice Prompt is set.',
		displayOptions: { show: { resource: ['tool'], operation: ['executeAndCall'] } },
	},
	choiceFixedCollection(['tool'], ['executeAndCall']),
	{
		displayName: 'Additional Fields',
		name: 'additionalFields',
		type: 'collection',
		placeholder: 'Add Field',
		default: {},
		displayOptions: { show: { resource: ['tool'], operation: ['executeAndCall'] } },
		options: [
			{
				displayName: 'Prefix',
				name: 'prefix',
				type: 'string',
				default: '',
				description: 'Text spoken before the tool result',
			},
			{
				displayName: 'Suffix',
				name: 'suffix',
				type: 'string',
				default: '',
				description: 'Text spoken after the tool result',
			},
			{
				displayName: 'Ring Timeout',
				name: 'ringTimeout',
				type: 'number',
				typeOptions: { minValue: 1, maxValue: 600 },
				default: 30,
				description: 'Seconds to let the phone ring before giving up',
			},
			{
				displayName: 'Call ID',
				name: 'callId',
				type: 'string',
				default: '',
				description: 'Custom call ID (letters, digits, ".", "_", "-"; max 64 chars)',
			},
			{
				displayName: 'Reformat for Speech',
				name: 'reformatForSpeech',
				type: 'boolean',
				default: false,
				description:
					'Whether the agent\'s LLM rewrites the composed message into natural spoken form without dropping any information',
			},
		],
	},
];

export async function executeTool(
	ctx: IExecuteFunctions,
	i: number,
	operation: string,
): Promise<IDataObject | IDataObject[]> {
	if (operation === 'getMany') {
		const response = await sipAgentApiRequest.call(ctx, 'GET', '/tools');
		return response as unknown as IDataObject[];
	}

	if (operation === 'get') {
		const toolName = ctx.getNodeParameter('toolName', i) as string;
		return await sipAgentApiRequest.call(
			ctx,
			'GET',
			`/tools/${encodeURIComponent(toolName)}`,
		);
	}

	if (operation === 'execute') {
		const toolName = ctx.getNodeParameter('toolName', i) as string;
		const params = parseJsonParameter(ctx, i, 'toolParams', 'Tool Parameters');
		const speakResult = ctx.getNodeParameter('speakResult', i, false) as boolean;
		const callId = ctx.getNodeParameter('callId', i, '') as string;
		const body: IDataObject = {
			params,
			speak_result: speakResult,
			...(callId ? { call_id: callId } : {}),
		};
		return await sipAgentApiRequest.call(
			ctx,
			'POST',
			`/tools/${encodeURIComponent(toolName)}/execute`,
			body,
		);
	}

	if (operation === 'executeAndCall') {
		const toolName = ctx.getNodeParameter('toolName', i) as string;
		const params = parseJsonParameter(ctx, i, 'toolParams', 'Tool Parameters');
		const extension = ctx.getNodeParameter('extension', i) as string;
		const callbackUrl = ctx.getNodeParameter('callbackUrl', i, '') as string;
		const choice = buildChoice(ctx, i);
		assertCallbackWithChoice(ctx, i, choice, callbackUrl);
		const additionalFields = ctx.getNodeParameter('additionalFields', i, {}) as IDataObject;
		const body: IDataObject = {
			params,
			extension,
			ring_timeout: (additionalFields.ringTimeout as number) ?? 30,
			...(additionalFields.prefix ? { prefix: additionalFields.prefix } : {}),
			...(additionalFields.suffix ? { suffix: additionalFields.suffix } : {}),
			...(additionalFields.callId ? { call_id: additionalFields.callId } : {}),
			...(additionalFields.reformatForSpeech ? { reformat_for_speech: true } : {}),
			...(callbackUrl ? { callback_url: callbackUrl } : {}),
			...(choice ? { choice } : {}),
		};
		// The API responds 200 with status "tool_failed" when the tool errors;
		// pass that through as-is so workflows can branch on it.
		return await sipAgentApiRequest.call(
			ctx,
			'POST',
			`/tools/${encodeURIComponent(toolName)}/call`,
			body,
		);
	}

	throw new NodeOperationError(ctx.getNode(), `Unknown operation "${operation}"`, {
		itemIndex: i,
	});
}
