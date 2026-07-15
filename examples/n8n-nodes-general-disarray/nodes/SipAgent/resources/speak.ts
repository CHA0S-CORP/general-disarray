import type { IDataObject, IExecuteFunctions, INodeProperties } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

import { sipAgentApiRequest, sipAgentApiUpload } from '../GenericFunctions';

export const speakProperties: INodeProperties[] = [
	{
		displayName: 'Operation',
		name: 'operation',
		type: 'options',
		noDataExpression: true,
		displayOptions: { show: { resource: ['speak'] } },
		options: [
			{
				name: 'Say',
				value: 'say',
				description: 'Speak a message into the active call',
				action: 'Say a message',
			},
			{
				name: 'Play Audio',
				value: 'play',
				description: 'Play an audio file from the item\'s binary data into the active call',
				action: 'Play an audio file',
			},
		],
		default: 'say',
	},
	{
		displayName: 'Message',
		name: 'message',
		type: 'string',
		required: true,
		default: '',
		description: 'The message to speak into the active call',
		displayOptions: { show: { resource: ['speak'], operation: ['say'] } },
	},
	{
		displayName: 'Input Binary Field',
		name: 'binaryPropertyName',
		type: 'string',
		required: true,
		default: 'data',
		description:
			'Name of the input item\'s binary field holding the audio file (WAV, FLAC, or OGG; MP3 when the agent\'s libsndfile supports it)',
		displayOptions: { show: { resource: ['speak'], operation: ['play'] } },
	},
	{
		displayName: 'Call ID',
		name: 'callId',
		type: 'string',
		default: '',
		description:
			'Optional call ID; when set it must match the currently active call, otherwise the request is rejected',
		displayOptions: { show: { resource: ['speak'], operation: ['say', 'play'] } },
	},
	{
		displayName: 'Reformat for Speech',
		name: 'reformatForSpeech',
		type: 'boolean',
		default: false,
		description:
			'Whether the agent\'s LLM rewrites the message into natural spoken form without dropping any information',
		displayOptions: { show: { resource: ['speak'], operation: ['say'] } },
	},
];

export async function executeSpeak(
	ctx: IExecuteFunctions,
	i: number,
	operation: string,
): Promise<IDataObject | IDataObject[]> {
	if (operation === 'say') {
		const message = ctx.getNodeParameter('message', i) as string;
		const callId = ctx.getNodeParameter('callId', i, '') as string;
		const reformat = ctx.getNodeParameter('reformatForSpeech', i, false) as boolean;

		// POST /speak takes query parameters only — no JSON body.
		return sipAgentApiRequest.call(ctx, 'POST', '/speak', undefined, {
			message,
			...(callId && { call_id: callId }),
			...(reformat && { reformat_for_speech: 'true' }),
		});
	}

	if (operation === 'play') {
		const binaryPropertyName = ctx.getNodeParameter('binaryPropertyName', i) as string;
		const callId = ctx.getNodeParameter('callId', i, '') as string;

		const binary = ctx.helpers.assertBinaryData(i, binaryPropertyName);
		const buffer = await ctx.helpers.getBinaryDataBuffer(i, binaryPropertyName);

		// POST /play takes the audio file bytes as the raw request body.
		return sipAgentApiUpload.call(
			ctx,
			'/play',
			buffer,
			binary.mimeType || 'application/octet-stream',
			callId ? { call_id: callId } : undefined,
		);
	}

	throw new NodeOperationError(ctx.getNode(), `Unknown operation "${operation}"`, {
		itemIndex: i,
	});
}
