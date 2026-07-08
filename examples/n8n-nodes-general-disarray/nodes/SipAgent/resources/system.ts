import type { IDataObject, IExecuteFunctions, INodeProperties } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

import { sipAgentApiRequest } from '../GenericFunctions';

export const systemProperties: INodeProperties[] = [
	{
		displayName: 'Operation',
		name: 'operation',
		type: 'options',
		noDataExpression: true,
		displayOptions: { show: { resource: ['system'] } },
		options: [
			{
				name: 'Health',
				value: 'health',
				description: 'Get the agent health status',
				action: 'Get health',
			},
			{
				name: 'Get Queue',
				value: 'getQueue',
				description: 'Get the outbound call queue status',
				action: 'Get queue',
			},
		],
		default: 'health',
	},
	{
		displayName: 'Deep',
		name: 'deep',
		type: 'boolean',
		default: false,
		description: 'Whether to also probe the vLLM, Speaches and Redis dependencies',
		displayOptions: { show: { resource: ['system'], operation: ['health'] } },
	},
];

export async function executeSystem(
	ctx: IExecuteFunctions,
	i: number,
	operation: string,
): Promise<IDataObject | IDataObject[]> {
	if (operation === 'health') {
		const deep = ctx.getNodeParameter('deep', i, false) as boolean;
		return await sipAgentApiRequest.call(
			ctx,
			'GET',
			'/health',
			undefined,
			deep ? { deep: true } : undefined,
		);
	}

	if (operation === 'getQueue') {
		return await sipAgentApiRequest.call(ctx, 'GET', '/queue');
	}

	throw new NodeOperationError(ctx.getNode(), `Unknown operation "${operation}"`, {
		itemIndex: i,
	});
}
