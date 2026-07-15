import type { IDataObject, IExecuteFunctions, INodeProperties } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

/**
 * The "Choice Prompt" UI, reused by Call:Make and Tool:Execute and Call.
 * Maps to the agent's ChoicePrompt schema (POST /call, POST /tools/{name}/call).
 */
export function choiceFixedCollection(resources: string[], operations: string[]): INodeProperties {
	return {
		displayName: 'Choice Prompt',
		name: 'choiceUi',
		type: 'fixedCollection',
		default: {},
		placeholder: 'Add Choice Prompt',
		description:
			'Ask the callee a question and collect a spoken answer or DTMF keypress. Requires a Callback URL — the agent POSTs the result there (use a SIP Agent Trigger node).',
		displayOptions: { show: { resource: resources, operation: operations } },
		options: [
			{
				displayName: 'Choice',
				name: 'choiceValues',
				values: [
					{
						displayName: 'Prompt',
						name: 'prompt',
						type: 'string',
						default: '',
						required: true,
						description: 'Question spoken to the callee, e.g. "Should I confirm the appointment?"',
					},
					{
						displayName: 'Options',
						name: 'options',
						type: 'fixedCollection',
						typeOptions: { multipleValues: true },
						default: {},
						placeholder: 'Add Option',
						options: [
							{
								displayName: 'Option',
								name: 'optionValues',
								values: [
									{
										displayName: 'Value',
										name: 'value',
										type: 'string',
										default: '',
										required: true,
										description: 'Canonical answer value, e.g. "yes"',
									},
									{
										displayName: 'Synonyms',
										name: 'synonyms',
										type: 'string',
										default: '',
										description: 'Comma-separated alternatives, e.g. "yeah, sure, ok"',
									},
									{
										displayName: 'DTMF Key',
										name: 'dtmf',
										type: 'string',
										default: '',
										description:
											'Single phone key 0-9, * or # that selects this option (defaults to the option\'s 1-based position)',
									},
								],
							},
						],
					},
					{
						displayName: 'Timeout (Seconds)',
						name: 'timeoutSeconds',
						type: 'number',
						typeOptions: { minValue: 1, maxValue: 300 },
						default: 30,
					},
					{
						displayName: 'Repeat Count',
						name: 'repeatCount',
						type: 'number',
						typeOptions: { minValue: 1, maxValue: 10 },
						default: 2,
					},
				],
			},
		],
	};
}

/**
 * Build the ChoicePrompt request body from the choiceUi fixedCollection.
 * Returns undefined when no choice prompt was configured.
 */
export function buildChoice(ctx: IExecuteFunctions, i: number): IDataObject | undefined {
	const ui = ctx.getNodeParameter('choiceUi', i, {}) as IDataObject;
	const cv = ui.choiceValues as IDataObject | undefined;
	if (!cv || !cv.prompt) return undefined;
	const optionValues = ((cv.options as IDataObject | undefined)?.optionValues ??
		[]) as IDataObject[];
	if (optionValues.length === 0) {
		throw new NodeOperationError(
			ctx.getNode(),
			'Choice Prompt needs at least one option',
			{ itemIndex: i },
		);
	}
	return {
		prompt: cv.prompt,
		options: optionValues.map((o) => ({
			value: o.value,
			synonyms: ((o.synonyms as string) || '')
				.split(',')
				.map((s) => s.trim())
				.filter(Boolean),
			...(o.dtmf ? { dtmf: o.dtmf } : {}),
		})),
		timeout_seconds: (cv.timeoutSeconds as number) ?? 30,
		repeat_count: (cv.repeatCount as number) ?? 2,
	};
}

/**
 * The agent rejects choice without callback_url with a 422; fail fast with a
 * clearer message instead.
 */
export function assertCallbackWithChoice(
	ctx: IExecuteFunctions,
	i: number,
	choice: IDataObject | undefined,
	callbackUrl: string,
): void {
	if (choice && !callbackUrl) {
		throw new NodeOperationError(
			ctx.getNode(),
			'Callback URL is required when a Choice Prompt is set',
			{ itemIndex: i },
		);
	}
}

/**
 * Parse a `json`-type node parameter that may arrive as a string or an object.
 */
export function parseJsonParameter(
	ctx: IExecuteFunctions,
	i: number,
	name: string,
	displayName: string,
): IDataObject {
	const raw = ctx.getNodeParameter(name, i, {}) as unknown;
	if (raw === null || raw === undefined || raw === '') return {};
	if (typeof raw === 'object') return raw as IDataObject;
	try {
		const parsed = JSON.parse(raw as string);
		if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
			throw new Error('not a JSON object');
		}
		return parsed as IDataObject;
	} catch {
		throw new NodeOperationError(
			ctx.getNode(),
			`${displayName} must be a valid JSON object`,
			{ itemIndex: i },
		);
	}
}
