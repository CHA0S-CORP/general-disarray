import type {
	IDataObject,
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';
import { NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';

import { callProperties, executeCall } from './resources/call';
import { scheduleProperties, executeSchedule } from './resources/schedule';
import { speakProperties, executeSpeak } from './resources/speak';
import { systemProperties, executeSystem } from './resources/system';
import { toolProperties, executeTool } from './resources/tool';
import { virtualNumberProperties, executeVirtualNumber } from './resources/virtualNumber';

export class SipAgent implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'SIP Agent',
		name: 'sipAgent',
		icon: 'file:sipAgent.svg',
		group: ['output'],
		version: 1,
		subtitle: '={{$parameter["operation"] + ": " + $parameter["resource"]}}',
		description: 'Interact with the General Disarray SIP AI phone assistant',
		defaults: {
			name: 'SIP Agent',
		},
		usableAsTool: true,
		inputs: [NodeConnectionTypes.Main],
		outputs: [NodeConnectionTypes.Main],
		credentials: [
			{
				name: 'sipAgentApi',
				required: true,
			},
		],
		properties: [
			{
				displayName: 'Resource',
				name: 'resource',
				type: 'options',
				noDataExpression: true,
				options: [
					{ name: 'Call', value: 'call' },
					{ name: 'Schedule', value: 'schedule' },
					{ name: 'Speak', value: 'speak' },
					{ name: 'System', value: 'system' },
					{ name: 'Tool', value: 'tool' },
					{ name: 'Virtual Number', value: 'virtualNumber' },
				],
				default: 'call',
			},
			...callProperties,
			...scheduleProperties,
			...speakProperties,
			...systemProperties,
			...toolProperties,
			...virtualNumberProperties,
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];
		const resource = this.getNodeParameter('resource', 0) as string;
		const operation = this.getNodeParameter('operation', 0) as string;

		for (let i = 0; i < items.length; i++) {
			try {
				let result: IDataObject | IDataObject[];
				switch (resource) {
					case 'call':
						result = await executeCall(this, i, operation);
						break;
					case 'schedule':
						result = await executeSchedule(this, i, operation);
						break;
					case 'speak':
						result = await executeSpeak(this, i, operation);
						break;
					case 'system':
						result = await executeSystem(this, i, operation);
						break;
					case 'tool':
						result = await executeTool(this, i, operation);
						break;
					case 'virtualNumber':
						result = await executeVirtualNumber(this, i, operation);
						break;
					default:
						throw new NodeOperationError(
							this.getNode(),
							`Unknown resource "${resource}"`,
							{ itemIndex: i },
						);
				}
				if (Array.isArray(result)) {
					for (const entry of result) {
						returnData.push({ json: entry, pairedItem: { item: i } });
					}
				} else {
					returnData.push({ json: result, pairedItem: { item: i } });
				}
			} catch (error) {
				if (this.continueOnFail()) {
					returnData.push({
						json: { error: (error as Error).message },
						pairedItem: { item: i },
					});
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}
}
