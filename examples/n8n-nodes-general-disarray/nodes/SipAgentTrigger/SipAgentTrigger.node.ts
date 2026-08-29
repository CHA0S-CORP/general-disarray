import { createHmac, timingSafeEqual } from 'crypto';

import type {
	IDataObject,
	IHookFunctions,
	INodeType,
	INodeTypeDescription,
	IWebhookFunctions,
	IWebhookResponseData,
} from 'n8n-workflow';
import { NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';

import { sipAgentApiRequest } from '../SipAgent/GenericFunctions';

/** Events a registered trigger number can fire (agent: virtual_numbers.VALID_EVENTS). */
const TRIGGER_NUMBER_EVENTS = ['answered', 'first_speech', 'speech', 'completed'];

export class SipAgentTrigger implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'SIP Agent Trigger',
		name: 'sipAgentTrigger',
		icon: 'file:sipAgentTrigger.svg',
		group: ['trigger'],
		version: 1,
		subtitle:
			'={{$parameter["mode"] === "triggerNumber" ? "Trigger number " + ($parameter["number"] || "(auto)") : "Webhook"}}',
		description:
			'Starts a workflow from a phone call: registers a trigger number on the General Disarray SIP AI phone assistant, or receives its call-lifecycle / choice-callback webhooks',
		defaults: {
			name: 'SIP Agent Trigger',
		},
		inputs: [],
		outputs: [NodeConnectionTypes.Main],
		credentials: [
			{
				name: 'sipAgentApi',
				// Optional for plain webhook mode; Trigger Number mode needs it to
				// register the number and fails with a clear error without it.
				required: false,
			},
		],
		webhooks: [
			{
				name: 'default',
				httpMethod: 'POST',
				responseMode: 'onReceived',
				path: 'webhook',
			},
		],
		properties: [
			{
				displayName: 'Mode',
				name: 'mode',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Trigger Number',
						value: 'triggerNumber',
						description:
							'Register a phone extension on the agent while this workflow is active; a call dialed to it starts the workflow',
					},
					{
						name: 'Webhook',
						value: 'webhook',
						description:
							'Receive call-lifecycle events (CALL_EVENT_WEBHOOK_URL) and choice/outcome callbacks the agent POSTs to this URL',
					},
				],
				// 'webhook' so nodes saved before this option existed keep their
				// passive behaviour (n8n applies the default to any unsaved param;
				// 'triggerNumber' here would silently try to register a number —
				// and fail without a credential — on every legacy workflow).
				default: 'webhook',
			},

			// ----------------------------------
			//         mode: triggerNumber
			// ----------------------------------
			{
				displayName: 'Number',
				name: 'number',
				type: 'string',
				default: '',
				placeholder: '7350',
				description:
					'Extension the agent listens on (digits, * or #). Leave empty to auto-allocate from the agent\'s VIRTUAL_NUMBER_RANGE; the allocated number is in the trigger payload and under GET /virtual-numbers.',
				displayOptions: { show: { mode: ['triggerNumber'] } },
			},
			{
				displayName: 'Trigger On',
				name: 'triggerEvents',
				type: 'multiOptions',
				options: [
					{
						name: 'Call Answered',
						value: 'answered',
						description: 'The call is matched to the number, before the greeting plays',
					},
					{
						name: 'First Utterance',
						value: 'first_speech',
						description:
							'The caller\'s first transcribed sentence (payload field "text") — listen for what they want',
					},
					{
						name: 'Every Utterance',
						value: 'speech',
						description: 'Every transcribed sentence the caller says during the call',
					},
					{
						name: 'Call Completed',
						value: 'completed',
						description: 'The call ended; includes the transcript unless disabled below',
					},
				],
				default: ['answered'],
				description:
					'Which moments of a call to this number start the workflow. Branch on {{$json.event}} (virtual_number.answered / .first_speech / .speech / .completed) when several are selected.',
				displayOptions: { show: { mode: ['triggerNumber'] } },
			},
			{
				displayName: 'Purpose',
				name: 'purpose',
				type: 'string',
				default: 'Calls to this number start an automation workflow.',
				description:
					'What this number is for — injected into the agent\'s system prompt so the assistant handles the conversation in context',
				displayOptions: { show: { mode: ['triggerNumber'] } },
			},
			{
				displayName: 'Options',
				name: 'triggerOptions',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				displayOptions: { show: { mode: ['triggerNumber'] } },
				options: [
					{
						displayName: 'Greeting',
						name: 'greeting',
						type: 'string',
						default: '',
						description: 'Custom greeting spoken instead of the agent\'s default one',
					},
					{
						displayName: 'Include Transcript',
						name: 'includeTranscript',
						type: 'boolean',
						default: true,
						description: 'Whether the Call Completed payload carries the full transcript',
					},
				],
			},

			// ----------------------------------
			//         mode: webhook
			// ----------------------------------
			{
				displayName: 'Events',
				name: 'events',
				type: 'multiOptions',
				options: [
					{
						name: 'Call Ended',
						value: 'call.ended',
						description: 'A call on the agent ended (inbound or outbound)',
					},
					{
						name: 'Call Started',
						value: 'call.started',
						description: 'A call on the agent started (inbound or outbound)',
					},
					{
						name: 'Choice Result / Call Outcome',
						value: 'choice.result',
						description:
							'The callback_url payload sent when an outbound call (with an optional choice prompt) completes — these payloads carry no "event" field',
					},
					{
						name: 'Virtual Number Events',
						value: 'virtual_number',
						description:
							'Payloads from virtual numbers created elsewhere (SIP Agent → Virtual Number → Create) whose callback URL points here',
					},
				],
				default: ['call.ended', 'call.started', 'choice.result', 'virtual_number'],
				description:
					'Which agent events start the workflow. Non-matching deliveries are acknowledged with 200 but do not start an execution.',
				displayOptions: { show: { mode: ['webhook'] } },
			},

			// ----------------------------------
			//         shared: signature check
			// ----------------------------------
			{
				displayName: 'Require Signature',
				name: 'requireSignature',
				type: 'boolean',
				default: false,
				description:
					'Whether to reject requests that lack a valid HMAC-SHA256 signature (X-Timestamp / X-Signature headers). Needs the Signing Secret set on the SIP Agent API credential, matching the agent\'s WEBHOOK_SIGNING_SECRET.',
			},
			{
				displayName: 'Tolerance (Seconds)',
				name: 'tolerance',
				type: 'number',
				typeOptions: { minValue: 1 },
				default: 300,
				description:
					'Maximum allowed age of the X-Timestamp header before a signed request is rejected as stale',
			},
		],
	};

	// Trigger Number mode: n8n calls these when the workflow is activated /
	// deactivated (and around "Listen for test event"), which is exactly when
	// the number should exist on the agent. The registered id lives in the
	// node's static data so deactivation can remove it.
	webhookMethods = {
		default: {
			async checkExists(this: IHookFunctions): Promise<boolean> {
				if (this.getNodeParameter('mode') !== 'triggerNumber') {
					// Plain webhook mode has nothing to register; n8n keeps the URL live.
					return true;
				}
				const staticData = this.getWorkflowStaticData('node');
				const id = staticData.virtualNumberId as string | undefined;
				if (!id) return false;
				try {
					const entry = await sipAgentApiRequest.call(
						this,
						'GET',
						`/virtual-numbers/${encodeURIComponent(id)}`,
					);
					// The webhook URL can change (test vs production); re-register
					// if it drifted so calls land on the right URL.
					const wantUrl = this.getNodeWebhookUrl('default');
					if (entry.persistent && (!entry.callback_url || entry.callback_url === wantUrl)) {
						return true;
					}
				} catch {
					// 404 (deleted / agent restarted without the file) → re-create.
				}
				delete staticData.virtualNumberId;
				return false;
			},

			async create(this: IHookFunctions): Promise<boolean> {
				if (this.getNodeParameter('mode') !== 'triggerNumber') return true;

				const webhookUrl = this.getNodeWebhookUrl('default');
				if (!webhookUrl) {
					throw new NodeOperationError(this.getNode(), 'Could not determine the webhook URL');
				}
				const events = (this.getNodeParameter('triggerEvents', ['answered']) as string[]).filter(
					(e) => TRIGGER_NUMBER_EVENTS.includes(e),
				);
				if (events.length === 0) {
					throw new NodeOperationError(
						this.getNode(),
						'Select at least one "Trigger On" event',
					);
				}
				const options = this.getNodeParameter('triggerOptions', {}) as IDataObject;
				const body: IDataObject = {
					purpose:
						(this.getNodeParameter('purpose', '') as string) ||
						'Calls to this number start an automation workflow.',
					persistent: true,
					events,
					callback_url: webhookUrl,
					include_transcript: options.includeTranscript !== false,
				};
				const number = ((this.getNodeParameter('number', '') as string) || '').trim();
				if (number) body.number = number;
				if (options.greeting) body.greeting = options.greeting;

				const entry = await sipAgentApiRequest.call(this, 'POST', '/virtual-numbers', body);
				const staticData = this.getWorkflowStaticData('node');
				staticData.virtualNumberId = entry.id;
				staticData.virtualNumber = entry.number;
				return true;
			},

			async delete(this: IHookFunctions): Promise<boolean> {
				if (this.getNodeParameter('mode') !== 'triggerNumber') return true;
				const staticData = this.getWorkflowStaticData('node');
				const id = staticData.virtualNumberId as string | undefined;
				if (id) {
					try {
						await sipAgentApiRequest.call(
							this,
							'DELETE',
							`/virtual-numbers/${encodeURIComponent(id)}`,
						);
					} catch {
						// Already gone (404) or agent unreachable — nothing more to do;
						// a persistent number that outlives us is listed under
						// GET /virtual-numbers and can be deleted by hand.
					}
				}
				delete staticData.virtualNumberId;
				delete staticData.virtualNumber;
				return true;
			},
		},
	};

	async webhook(this: IWebhookFunctions): Promise<IWebhookResponseData> {
		const mode = this.getNodeParameter('mode', 'webhook') as string;
		const requireSignature = this.getNodeParameter('requireSignature', false) as boolean;
		const tolerance = this.getNodeParameter('tolerance', 300) as number;

		const credentials = await this.getCredentials('sipAgentApi').catch(() => undefined);
		const secret = ((credentials?.signingSecret as string) || '').trim();

		const headers = this.getHeaderData() as IDataObject;
		const signatureHeader = (headers['x-signature'] as string) || '';

		const reject = (message: string): IWebhookResponseData => {
			const res = this.getResponseObject();
			res.status(401).json({ error: message });
			return { noWebhookResponse: true };
		};

		if (requireSignature || (secret && signatureHeader)) {
			if (!secret) {
				return reject('Signature required but no signing secret is configured');
			}
			const timestamp = (headers['x-timestamp'] as string) || '';
			if (!timestamp || !signatureHeader) {
				return reject('Missing X-Timestamp or X-Signature header');
			}
			const ts = Number(timestamp);
			// Number('') / Number('abc') are NaN and every comparison with NaN
			// is false — check finiteness explicitly so garbage can't skip the
			// freshness gate.
			if (!Number.isFinite(ts) || Math.abs(Date.now() / 1000 - ts) > tolerance) {
				return reject('Stale timestamp');
			}
			const provided = signatureHeader.replace(/^sha256=/, '');
			const req = this.getRequestObject();
			const raw: Buffer =
				(req as unknown as { rawBody?: Buffer }).rawBody ??
				Buffer.from(JSON.stringify(req.body));
			const expected = createHmac('sha256', secret)
				.update(`${timestamp}.`)
				.update(raw)
				.digest();
			let providedBuffer: Buffer;
			try {
				providedBuffer = Buffer.from(provided, 'hex');
			} catch {
				return reject('Invalid signature');
			}
			if (
				providedBuffer.length !== expected.length ||
				!timingSafeEqual(providedBuffer, expected)
			) {
				return reject('Invalid signature');
			}
		}

		const body = this.getBodyData() as IDataObject;
		// Legacy choice/outcome callbacks carry no `event` field.
		const eventType =
			typeof body.event === 'string' && body.event !== '' ? (body.event as string) : 'choice.result';

		const ignore = (): IWebhookResponseData => {
			// Acknowledge with 200 so the agent's deliver_webhook doesn't retry,
			// but start no execution.
			const res = this.getResponseObject();
			res.status(200).json({ ok: true, ignored: eventType });
			return { noWebhookResponse: true };
		};

		if (mode === 'triggerNumber') {
			// The agent only sends the events we subscribed to; anything else
			// on this URL is noise (e.g. a stale CALL_EVENT_WEBHOOK_URL). The
			// number's "expired" lifecycle event can't occur for a persistent
			// number, but tolerate it the same way.
			if (!eventType.startsWith('virtual_number.') || eventType === 'virtual_number.expired') {
				return ignore();
			}
			const staticData = this.getWorkflowStaticData('node');
			if (staticData.virtualNumberId && body.id && body.id !== staticData.virtualNumberId) {
				return ignore();
			}
		} else {
			const events = this.getNodeParameter('events', [
				'call.ended',
				'call.started',
				'choice.result',
				'virtual_number',
			]) as string[];
			const key = eventType.startsWith('virtual_number.') ? 'virtual_number' : eventType;
			if (!events.includes(key)) {
				return ignore();
			}
		}

		return {
			workflowData: [this.helpers.returnJsonArray(body)],
		};
	}
}
