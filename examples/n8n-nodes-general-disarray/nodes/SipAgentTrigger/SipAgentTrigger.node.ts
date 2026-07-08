import { createHmac, timingSafeEqual } from 'crypto';

import type {
	IDataObject,
	INodeType,
	INodeTypeDescription,
	IWebhookFunctions,
	IWebhookResponseData,
} from 'n8n-workflow';
import { NodeConnectionTypes } from 'n8n-workflow';

export class SipAgentTrigger implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'SIP Agent Trigger',
		name: 'sipAgentTrigger',
		icon: 'file:sipAgentTrigger.svg',
		group: ['trigger'],
		version: 1,
		description:
			'Receives choice-callback webhooks from the General Disarray SIP AI phone assistant',
		defaults: {
			name: 'SIP Agent Trigger',
		},
		inputs: [],
		outputs: [NodeConnectionTypes.Main],
		credentials: [
			{
				name: 'sipAgentApi',
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

	async webhook(this: IWebhookFunctions): Promise<IWebhookResponseData> {
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
			if (Math.abs(Date.now() / 1000 - Number(timestamp)) > tolerance) {
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

		return {
			workflowData: [this.helpers.returnJsonArray(this.getBodyData() as IDataObject)],
		};
	}
}
