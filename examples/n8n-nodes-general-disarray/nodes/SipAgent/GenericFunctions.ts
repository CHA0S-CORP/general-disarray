import type {
	IDataObject,
	IExecuteFunctions,
	IHookFunctions,
	IHttpRequestMethods,
	ILoadOptionsFunctions,
	JsonObject,
} from 'n8n-workflow';
import { NodeApiError } from 'n8n-workflow';

/**
 * Make a request to the SIP agent REST API using the sipAgentApi credential.
 * Injects the API token as X-API-Key or Authorization: Bearer per the credential's
 * headerStyle, only when a token is set (auth is optional on the agent).
 */
export async function sipAgentApiRequest(
	this: IExecuteFunctions | ILoadOptionsFunctions | IHookFunctions,
	method: IHttpRequestMethods,
	endpoint: string,
	body?: IDataObject,
	qs?: IDataObject,
): Promise<IDataObject> {
	const credentials = await this.getCredentials('sipAgentApi');
	const headers: IDataObject = {};
	const token = (credentials.apiToken as string) || '';
	if (token) {
		if (credentials.headerStyle === 'bearer') {
			headers.Authorization = `Bearer ${token}`;
		} else {
			headers['X-API-Key'] = token;
		}
	}
	const baseUrl = (credentials.baseUrl as string).replace(/\/+$/, '');
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
		throw new NodeApiError(this.getNode(), error as JsonObject);
	}
}
