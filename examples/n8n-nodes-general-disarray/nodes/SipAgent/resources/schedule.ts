import type { IDataObject, IExecuteFunctions, INodeProperties } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

import { sipAgentApiRequest } from '../GenericFunctions';
import { parseJsonParameter } from '../shared';

export const scheduleProperties: INodeProperties[] = [
	{
		displayName: 'Operation',
		name: 'operation',
		type: 'options',
		noDataExpression: true,
		displayOptions: { show: { resource: ['schedule'] } },
		options: [
			{
				name: 'Create',
				value: 'create',
				description: 'Schedule an outbound call for later',
				action: 'Create a scheduled call',
			},
			{
				name: 'Get Many',
				value: 'getMany',
				description: 'List all scheduled calls',
				action: 'Get many scheduled calls',
			},
			{
				name: 'Get',
				value: 'get',
				description: 'Get a scheduled call by ID',
				action: 'Get a scheduled call',
			},
			{
				name: 'Delete',
				value: 'delete',
				description: 'Cancel a scheduled call',
				action: 'Delete a scheduled call',
			},
		],
		default: 'create',
	},

	// ----------------------------------
	//         schedule:create
	// ----------------------------------
	{
		displayName: 'Extension',
		name: 'extension',
		type: 'string',
		required: true,
		default: '',
		description: 'SIP extension or phone number to call',
		displayOptions: { show: { resource: ['schedule'], operation: ['create'] } },
	},
	{
		displayName: 'Content',
		name: 'contentMode',
		type: 'options',
		noDataExpression: true,
		options: [
			{
				name: 'Message',
				value: 'message',
				description: 'Speak a fixed message',
			},
			{
				name: 'Tool',
				value: 'tool',
				description: 'Run a tool at call time and speak its result',
			},
		],
		default: 'message',
		displayOptions: { show: { resource: ['schedule'], operation: ['create'] } },
	},
	{
		displayName: 'Message',
		name: 'message',
		type: 'string',
		required: true,
		default: '',
		description: 'Message to speak when the call is answered',
		displayOptions: {
			show: { resource: ['schedule'], operation: ['create'], contentMode: ['message'] },
		},
	},
	{
		displayName: 'Tool Name',
		name: 'toolName',
		type: 'string',
		required: true,
		default: '',
		placeholder: 'WEATHER',
		description:
			'Tool to execute when the schedule fires; its result is spoken on the call. Tool names are uppercase, e.g. WEATHER, TIMER (the server uppercases anyway).',
		displayOptions: {
			show: { resource: ['schedule'], operation: ['create'], contentMode: ['tool'] },
		},
	},
	{
		displayName: 'Tool Parameters',
		name: 'toolParams',
		type: 'json',
		default: '{}',
		description: 'Parameters passed to the tool, as a JSON object',
		displayOptions: {
			show: { resource: ['schedule'], operation: ['create'], contentMode: ['tool'] },
		},
	},
	{
		displayName: 'Schedule Type',
		name: 'scheduleType',
		type: 'options',
		noDataExpression: true,
		options: [
			{
				name: 'Delay',
				value: 'delay',
				description: 'Call after a number of seconds',
			},
			{
				name: 'At Time',
				value: 'atTime',
				description: 'Call at a specific time',
			},
		],
		default: 'delay',
		displayOptions: { show: { resource: ['schedule'], operation: ['create'] } },
	},
	{
		displayName: 'Delay (Seconds)',
		name: 'delaySeconds',
		type: 'number',
		typeOptions: { minValue: 1 },
		default: 60,
		description: 'Seconds from now until the call is made',
		displayOptions: {
			show: { resource: ['schedule'], operation: ['create'], scheduleType: ['delay'] },
		},
	},
	{
		displayName: 'At Time',
		name: 'atTime',
		type: 'string',
		required: true,
		default: '',
		placeholder: '15:30',
		description: 'Time of day (e.g. "15:30") or an ISO timestamp',
		displayOptions: {
			show: { resource: ['schedule'], operation: ['create'], scheduleType: ['atTime'] },
		},
	},
	{
		displayName: 'Additional Fields',
		name: 'additionalFields',
		type: 'collection',
		placeholder: 'Add Field',
		default: {},
		displayOptions: { show: { resource: ['schedule'], operation: ['create'] } },
		options: [
			{
				displayName: 'Timezone',
				name: 'timezone',
				type: 'string',
				default: 'America/Los_Angeles',
				description: 'IANA timezone used to interpret At Time',
			},
			{
				displayName: 'Prefix',
				name: 'prefix',
				type: 'string',
				default: '',
				description: 'Text spoken before the message or tool result',
			},
			{
				displayName: 'Suffix',
				name: 'suffix',
				type: 'string',
				default: '',
				description: 'Text spoken after the message or tool result',
			},
			{
				displayName: 'Callback URL',
				name: 'callbackUrl',
				type: 'string',
				default: '',
				description: 'URL the agent POSTs the call result to',
			},
			{
				displayName: 'Recurring',
				name: 'recurring',
				type: 'options',
				options: [
					{ name: 'None', value: '' },
					{ name: 'Daily', value: 'daily' },
					{ name: 'Weekdays', value: 'weekdays' },
					{ name: 'Weekends', value: 'weekends' },
					{ name: 'Cron Expression', value: 'cron' },
				],
				default: '',
				description: 'Repeat the call on a schedule instead of firing once',
			},
			{
				displayName: 'Cron Expression',
				name: 'cronExpression',
				type: 'string',
				default: '',
				placeholder: '30 8 * * 1-5',
				description: 'Standard 5-field cron expression (minute hour day month weekday)',
				displayOptions: { show: { recurring: ['cron'] } },
			},
		],
	},

	// ----------------------------------
	//         schedule:get / delete
	// ----------------------------------
	{
		displayName: 'Schedule ID',
		name: 'scheduleId',
		type: 'string',
		required: true,
		default: '',
		description: 'ID returned when the schedule was created',
		displayOptions: { show: { resource: ['schedule'], operation: ['get', 'delete'] } },
	},
];

export async function executeSchedule(
	ctx: IExecuteFunctions,
	i: number,
	operation: string,
): Promise<IDataObject | IDataObject[]> {
	if (operation === 'create') {
		const extension = ctx.getNodeParameter('extension', i) as string;
		const body: IDataObject = { extension };

		const contentMode = ctx.getNodeParameter('contentMode', i) as string;
		if (contentMode === 'tool') {
			const toolName = ctx.getNodeParameter('toolName', i) as string;
			if (!toolName) {
				throw new NodeOperationError(ctx.getNode(), 'Tool Name is required', { itemIndex: i });
			}
			body.tool = toolName;
			const toolParams = parseJsonParameter(ctx, i, 'toolParams', 'Tool Parameters');
			if (Object.keys(toolParams).length > 0) {
				body.tool_params = toolParams;
			}
		} else {
			const message = ctx.getNodeParameter('message', i) as string;
			if (!message) {
				throw new NodeOperationError(ctx.getNode(), 'Message is required', { itemIndex: i });
			}
			body.message = message;
		}

		const scheduleType = ctx.getNodeParameter('scheduleType', i) as string;
		if (scheduleType === 'atTime') {
			const atTime = ctx.getNodeParameter('atTime', i) as string;
			if (!atTime) {
				throw new NodeOperationError(ctx.getNode(), 'At Time is required', { itemIndex: i });
			}
			body.at_time = atTime;
		} else {
			body.delay_seconds = ctx.getNodeParameter('delaySeconds', i) as number;
		}

		const additionalFields = ctx.getNodeParameter('additionalFields', i, {}) as IDataObject;
		if (additionalFields.timezone) {
			body.timezone = additionalFields.timezone;
		}
		if (additionalFields.prefix) {
			body.prefix = additionalFields.prefix;
		}
		if (additionalFields.suffix) {
			body.suffix = additionalFields.suffix;
		}
		if (additionalFields.callbackUrl) {
			body.callback_url = additionalFields.callbackUrl;
		}
		const recurring = (additionalFields.recurring as string) || '';
		if (recurring === 'cron') {
			const cronExpression = ((additionalFields.cronExpression as string) || '').trim();
			if (!cronExpression) {
				throw new NodeOperationError(
					ctx.getNode(),
					'Cron Expression is required when Recurring is set to Cron Expression',
					{ itemIndex: i },
				);
			}
			body.recurring = cronExpression;
		} else if (recurring) {
			body.recurring = recurring;
		}

		return sipAgentApiRequest.call(ctx, 'POST', '/schedule', body);
	}

	if (operation === 'getMany') {
		const response = await sipAgentApiRequest.call(ctx, 'GET', '/schedule');
		return response as unknown as IDataObject[];
	}

	if (operation === 'get') {
		const scheduleId = ctx.getNodeParameter('scheduleId', i) as string;
		return sipAgentApiRequest.call(ctx, 'GET', `/schedule/${encodeURIComponent(scheduleId)}`);
	}

	if (operation === 'delete') {
		const scheduleId = ctx.getNodeParameter('scheduleId', i) as string;
		return sipAgentApiRequest.call(ctx, 'DELETE', `/schedule/${encodeURIComponent(scheduleId)}`);
	}

	throw new NodeOperationError(ctx.getNode(), `Unknown operation "${operation}"`, {
		itemIndex: i,
	});
}
