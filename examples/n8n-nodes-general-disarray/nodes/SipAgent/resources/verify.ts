import type { IDataObject, IExecuteFunctions, INodeProperties } from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

import { sipAgentApiRequest } from '../GenericFunctions';

export const verifyProperties: INodeProperties[] = [
	{
		displayName: 'Operation',
		name: 'operation',
		type: 'options',
		noDataExpression: true,
		displayOptions: { show: { resource: ['verify'] } },
		options: [
			{
				name: 'Verify Code',
				value: 'verify',
				description: 'Check a caller\'s PIN and/or one-time code out-of-band',
				action: 'Verify a code',
			},
			{
				name: 'Set Credentials',
				value: 'setCredentials',
				description: 'Enroll or update a caller\'s PIN and/or TOTP secret',
				action: 'Set caller credentials',
			},
			{
				name: 'Get Credentials',
				value: 'getCredentials',
				description: 'Get a caller\'s enrollment metadata (never the secret or PIN)',
				action: 'Get caller credentials',
			},
			{
				name: 'Delete Credentials',
				value: 'deleteCredentials',
				description: 'Remove a caller\'s enrolled credentials',
				action: 'Delete caller credentials',
			},
			{
				name: 'Get Current OTP',
				value: 'getOtp',
				description: 'Get the caller\'s current TOTP code (for delivery/testing)',
				action: 'Get current OTP',
			},
		],
		default: 'verify',
	},

	// Caller ID is required by every operation.
	{
		displayName: 'Caller ID',
		name: 'callerId',
		type: 'string',
		required: true,
		default: '',
		description:
			'Caller id (the user part of the SIP URI, e.g. "1001"). For Get Current OTP, the reserved id "global" returns the code for the global VERIFY_TOTP_SECRET.',
		displayOptions: { show: { resource: ['verify'] } },
	},

	// ----------------------------------
	//         verify:verify / setCredentials
	// ----------------------------------
	// For verify: the PIN to check. For setCredentials: the PIN to set/rotate.
	{
		displayName: 'PIN',
		name: 'pin',
		type: 'string',
		typeOptions: { password: true },
		default: '',
		description:
			'Static PIN — the value to check (Verify) or to set/rotate (Set Credentials). Leave empty to skip.',
		displayOptions: { show: { resource: ['verify'], operation: ['verify', 'setCredentials'] } },
	},

	// ----------------------------------
	//         verify:verify
	// ----------------------------------
	{
		displayName: 'OTP',
		name: 'otp',
		type: 'string',
		default: '',
		description: 'One-time (TOTP) code to check (leave empty to check only the PIN)',
		displayOptions: { show: { resource: ['verify'], operation: ['verify'] } },
	},

	// ----------------------------------
	//         verify:setCredentials
	// ----------------------------------
	{
		displayName: 'Generate TOTP Secret',
		name: 'generateTotp',
		type: 'boolean',
		default: false,
		description:
			'Whether to mint a fresh random TOTP secret for this caller (the response includes a provisioning URI to import into an authenticator app)',
		displayOptions: { show: { resource: ['verify'], operation: ['setCredentials'] } },
	},
	{
		displayName: 'TOTP Secret',
		name: 'totpSecret',
		type: 'string',
		typeOptions: { password: true },
		default: '',
		description:
			'Existing base32 TOTP secret to store (ignored when Generate TOTP Secret is on)',
		displayOptions: {
			show: { resource: ['verify'], operation: ['setCredentials'], generateTotp: [false] },
		},
	},
];

export async function executeVerify(
	ctx: IExecuteFunctions,
	i: number,
	operation: string,
): Promise<IDataObject | IDataObject[]> {
	const callerId = ctx.getNodeParameter('callerId', i) as string;
	if (!callerId) {
		throw new NodeOperationError(ctx.getNode(), 'Caller ID is required', { itemIndex: i });
	}

	if (operation === 'verify') {
		const pin = ctx.getNodeParameter('pin', i, '') as string;
		const otp = ctx.getNodeParameter('otp', i, '') as string;
		if (!pin && !otp) {
			throw new NodeOperationError(ctx.getNode(), 'Provide a PIN and/or an OTP to check', {
				itemIndex: i,
			});
		}
		const body: IDataObject = { caller_id: callerId };
		if (pin) body.pin = pin;
		if (otp) body.otp = otp;
		return sipAgentApiRequest.call(ctx, 'POST', '/verify', body);
	}

	if (operation === 'setCredentials') {
		const pin = ctx.getNodeParameter('pin', i, '') as string;
		const generateTotp = ctx.getNodeParameter('generateTotp', i, false) as boolean;
		const totpSecret = generateTotp
			? ''
			: (ctx.getNodeParameter('totpSecret', i, '') as string);
		if (!pin && !totpSecret && !generateTotp) {
			throw new NodeOperationError(
				ctx.getNode(),
				'Provide a PIN, a TOTP secret, or enable Generate TOTP Secret',
				{ itemIndex: i },
			);
		}
		const body: IDataObject = { caller_id: callerId };
		if (pin) body.pin = pin;
		if (totpSecret) body.totp_secret = totpSecret;
		if (generateTotp) body.generate_totp = true;
		return sipAgentApiRequest.call(ctx, 'POST', '/verify/credentials', body);
	}

	if (operation === 'getCredentials') {
		return sipAgentApiRequest.call(
			ctx,
			'GET',
			`/verify/credentials/${encodeURIComponent(callerId)}`,
		);
	}

	if (operation === 'deleteCredentials') {
		return sipAgentApiRequest.call(
			ctx,
			'DELETE',
			`/verify/credentials/${encodeURIComponent(callerId)}`,
		);
	}

	if (operation === 'getOtp') {
		return sipAgentApiRequest.call(ctx, 'GET', `/verify/otp/${encodeURIComponent(callerId)}`);
	}

	throw new NodeOperationError(ctx.getNode(), `Unknown operation "${operation}"`, {
		itemIndex: i,
	});
}
