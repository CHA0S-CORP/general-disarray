import type { IDataObject, IExecuteFunctions, INodeProperties } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

import { sipAgentApiRequest } from '../GenericFunctions';

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
		displayName: 'Call ID',
		name: 'callId',
		type: 'string',
		default: '',
		description:
			'Optional call ID; when set it must match the currently active call, otherwise the request is rejected',
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

		// POST /speak takes query parameters only — no JSON body.
		return sipAgentApiRequest.call(ctx, 'POST', '/speak', undefined, {
			message,
			...(callId && { call_id: callId }),
		});
	}

	throw new NodeOperationError(ctx.getNode(), `Unknown operation "${operation}"`, {
		itemIndex: i,
	});
}
