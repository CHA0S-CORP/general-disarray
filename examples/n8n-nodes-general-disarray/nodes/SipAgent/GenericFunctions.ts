import type {
	IDataObject,
	IExecuteFunctions,
	IHookFunctions,
	IHttpRequestMethods,
	ILoadOptionsFunctions,
	JsonObject,
} from 'n8n-workflow';
import { NodeApiError } from 'n8n-workflow';

type SipAgentContext = IExecuteFunctions | ILoadOptionsFunctions | IHookFunctions;

/** Auth headers + base URL from the sipAgentApi credential. */
async function resolveCredentials(
	ctx: SipAgentContext,
): Promise<{ baseUrl: string; headers: IDataObject }> {
	const credentials = await ctx.getCredentials('sipAgentApi');
	const headers: IDataObject = {};
	const token = (credentials.apiToken as string) || '';
	if (token) {
		if (credentials.headerStyle === 'bearer') {
			headers.Authorization = `Bearer ${token}`;
		} else {
			headers['X-API-Key'] = token;
		}
	}
	return { baseUrl: (credentials.baseUrl as string).replace(/\/+$/, ''), headers };
}

/**
 * Surface the agent's FastAPI error detail as the headline message instead
 * of n8n's generic HTTP-status text (e.g. "No active call to speak to"
 * rather than "The resource you are requesting could not be found").
 */
function toNodeApiError(ctx: SipAgentContext, error: unknown): NodeApiError {
	const detail = (error as { response?: { data?: { detail?: unknown } } }).response?.data
		?.detail;
	let message: string | undefined;
	if (typeof detail === 'string' && detail) {
		message = detail;
	} else if (Array.isArray(detail)) {
		// FastAPI 422 validation errors: [{loc, msg, type}, ...]
		message = detail
			.map((d) => (d && typeof d === 'object' && 'msg' in d ? String(d.msg) : JSON.stringify(d)))
			.join('; ');
	}
	return new NodeApiError(ctx.getNode(), error as JsonObject, message ? { message } : undefined);
}

/**
 * Make a request to the SIP agent REST API using the sipAgentApi credential.
 * Injects the API token as X-API-Key or Authorization: Bearer per the credential's
 * headerStyle, only when a token is set (auth is optional on the agent).
 */
export async function sipAgentApiRequest(
	this: SipAgentContext,
	method: IHttpRequestMethods,
	endpoint: string,
	body?: IDataObject,
	qs?: IDataObject,
): Promise<IDataObject> {
	const { baseUrl, headers } = await resolveCredentials(this);
	try {
		return (await this.helpers.httpRequest({
			method,
			url: `${baseUrl}${endpoint}`,
			headers,
			body,
			qs,
			json: true,
		})) as IDataObject;
	} catch (error) {
		throw toNodeApiError(this, error);
	}
}

/**
 * POST a raw binary body (e.g. an audio file for /play) to the SIP agent API.
 * Same credential/auth handling as sipAgentApiRequest, but no JSON encoding.
 */
export async function sipAgentApiUpload(
	this: IExecuteFunctions,
	endpoint: string,
	data: Buffer,
	contentType: string,
	qs?: IDataObject,
): Promise<IDataObject> {
	const { baseUrl, headers } = await resolveCredentials(this);
	try {
		// json:false so the Buffer body goes over the wire untouched; the
		// agent still answers JSON, so parse the response text ourselves.
		const response = await this.helpers.httpRequest({
			method: 'POST',
			url: `${baseUrl}${endpoint}`,
			headers: { ...headers, 'Content-Type': contentType || 'application/octet-stream' },
			body: data,
			qs,
			json: false,
		});
		if (typeof response === 'string') {
			try {
				return JSON.parse(response) as IDataObject;
			} catch {
				return { response };
			}
		}
		return response as IDataObject;
	} catch (error) {
		throw toNodeApiError(this, error);
	}
}
