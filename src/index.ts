import { Bot, Context, webhookCallback, GrammyError, HttpError } from "grammy";
import { GoogleGenAI, createUserContent, createModelContent} from "@google/genai";

import telegramifyMarkdown from "telegramify-markdown"
//@ts-ignore
import { extractAllOGInfo } from "./og"

type MessagePart = string;
interface TextPart {
    text: string;
}

export interface Env {
	DB: D1Database;
	GEMINI_API_KEY: string;
	SECRET_TELEGRAM_API_TOKEN: string;
	account_id: string; // CF AI Gateway
}

interface MyContext extends Context {
	env: Env;
}

function dispatchContent(content: string): { type: "text", text: string } | { type: "image_url", image_url: { url: string } } {
	if (content.startsWith("data:image/jpeg;base64,")) {
		return ({
			"type": "image_url",
			"image_url": {
				"url": content
			},
		})
	}
	return ({
		"type": "text",
		"text": content,
	});
}

function getMessageLink(r: { groupId: string, messageId: number }) {
	return `https://t.me/c/${parseInt(r.groupId.slice(2))}/${r.messageId}`;
}

function getSendTime(r: R) {
	return new Date(r.timeStamp).toLocaleString("zh-CN", { timeZone: "Asia/Shanghai" });
}

function escapeMarkdownV2(text: string) {
	// 注意：反斜杠 \ 本身也需要转义，所以正则表达式中是 \\\\
	// 或者直接在字符串中使用 \
	const reservedChars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'];
	// 正则表达式需要转义特殊字符
	const escapedChars = reservedChars.map(char => '\\' + char).join('');
	const regex = new RegExp(`([${escapedChars}])`, 'g');
	return text.replace(regex, '\\$1');
}

/**
 * 将数字转换为上标数字
 * @param {number} num - 要转换的数字
 * @returns {string} 上标形式的数字
 */
export function toSuperscript(num: number) {
	const superscripts = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'};
	return num.toString().split('').map(digit => superscripts[digit as keyof typeof superscripts]).join('');
}
/**
 * 处理 Markdown 文本中的重复链接，将其转换为顺序编号的格式
 * @param {string} text - 输入的 Markdown 文本
 * @param {Object} options - 配置选项
 * @param {string} options.prefix - 链接文本的前缀，默认为"链接"
 * @param {boolean} options.useEnglish - 是否使用英文(link1)而不是中文(链接1)，默认为 false
 * @returns {string} 处理后的 Markdown 文本
 */
export function processMarkdownLinks(text: string, options: { prefix: string, useEnglish: boolean } = {
	prefix: '引用',
	useEnglish: false
}) {
	const {
		prefix,
		useEnglish
	} = options;

	// 用于存储已经出现过的链接
	const linkMap = new Map();
	let linkCounter = 1;

	// 匹配 markdown 链接的正则表达式
	const linkPattern = /\[([^\]]+)\]\(([^)]+)\)/g;

	return text.replace(linkPattern, (match, displayText, url) => {
		// 只处理显示文本和 URL 完全相同的情况
		if (displayText !== url) {
			return match; // 保持原样
		}

		// 如果这个 URL 已经出现过，使用已存在的编号
		if (!linkMap.has(url)) {
			linkMap.set(url, linkCounter++);
		}
		const linkNumber = linkMap.get(url);

		// 根据选项决定使用中文还是英文格式
		const linkPrefix = useEnglish ? 'link' : prefix;

		// 返回新的格式 [链接1](原URL) 或 [link1](原URL)
		return `[${linkPrefix}${toSuperscript(linkNumber)}](${url})`;
	});
}

type R = {
	groupId: string;
	userName: string;
	content: string;
	messageId: number;
	timeStamp: number;
}
const model = "gemini-2.0-flash";
const reasoning_effort = "none";
const temperature = 0.4;
function getGenModel(env: Env) {
	// const openai = new OpenAI({
	// 	apiKey: env.GEMINI_API_KEY,
	// 	baseURL: "https://generativelanguage.googleapis.com/v1beta/openai/",
	// 	timeout: 999999999999,
	// });
	const googleGenAI = new GoogleGenAI({ apiKey: env.GEMINI_API_KEY });
	// const account_id = env.account_id; For Cloudflare AI Gateway, temporarily not used
	return googleGenAI;
}

function foldText(text: string) {
	return '**>' + text.split("\n").map((line) => '>' + line).join("\n") + '||';
}

// System prompts for different scenarios
const SYSTEM_PROMPTS = {
	summarizeChat: `你是一个专业的群聊概括助手。你的任务是用符合群聊风格的语气概括对话内容。
对话将按以下格式提供：
====================
用户名:
发言内容
相应链接
====================

请遵循以下指南：
1. 如果对话包含多个主题，请分条概括
2. 如果对话中提到图片，请在概括中包含相关内容描述
3. 在回答中用markdown格式引用原对话的链接
4. 链接格式应为：[引用1](链接本体)、[关键字1](链接本体)等
5. 概括要简洁明了，捕捉对话的主要内容和情绪
6. 概括的开头使用"本日群聊总结如下："`,

	answerQuestion: `你是一个群聊智能助手。你的任务是基于提供的群聊记录回答用户的问题。
群聊记录将按以下格式提供：
====================
用户名:
发言内容
相应链接
====================

请遵循以下指南：
1. 用符合群聊风格的语气回答问题
2. 在回答中引用相关的原始消息作为依据
3. 使用markdown格式引用原对话，格式为：[引用1](链接本体)、[关键字1](链接本体)
4. 在链接两侧添加空格
5. 如果找不到相关信息，请诚实说明
6. 回答应该简洁但内容完整`
};

function getCommandVar(str: string, delim: string) {
	return str.slice(str.indexOf(delim) + delim.length);
}

function messageTemplate(s: string) {
	return `下面由免费 ${escapeMarkdownV2(model)} 概括群聊信息\n` + s;
}
/**
 *
 * @param text
 * @description I dont know why, but llm keep output tme.cat, so we need to fix it
 * @returns
 */
function fixLink(text: string) {
	return text.replace(/tme\.cat/g, "t.me/c").replace(/\/c\/c/g, "/c");
}
function getUserName(msg: any) {
	if (msg?.sender_chat?.title) {
		return msg.sender_chat.title as string;
	}
	return msg.from?.first_name as string || "anonymous";
}

async function generateAndSendSummary(groupId: string | number, env: Env): Promise<void> {
    try {
        console.log(`[summary-job] Processing group: ${groupId}`);

        // 1. 获取该群组过去24小时的消息
        const { results } = await env.DB.prepare(
            'SELECT id, timeStamp, userName, content, messageId, groupName FROM Messages WHERE groupId=? AND timeStamp >= ? ORDER BY timeStamp ASC'
        ).bind(groupId, Date.now() - 24 * 60 * 60 * 1000).all<R>();

        if (!results || results.length < 10) { // 如果消息太少，则跳过
            console.log(`[summary-job] Skipping group ${groupId} due to insufficient messages (${results?.length ?? 0}).`);
            return;
        }

        // 2. 构建结构化的 LLM 输入 (与 ask/summary 命令完全一致)
        const userContentParts: TextPart[] = results.flatMap((r: R) => [
            { text: `${r.userName}:` },
            { text: r.content },
            { text: getMessageLink(r) }
        ]);

        // 3. 调用 Google Generative AI API (使用标准的 generateContent)
        const result = await getGenModel(env).models.generateContent({
            model,
            contents: [
                createModelContent(SYSTEM_PROMPTS.summarizeChat),
                createUserContent(userContentParts),
            ],
            config: {
                temperature: 0.5,
                maxOutputTokens: 4096,
            },
        });

        const summaryText = result.text;
        if (!summaryText) {
            console.error(`[summary-job] Failed to generate summary for group ${groupId}. LLM returned empty response.`);
            return;
        }

        // 4. 格式化并发送消息
        const formattedReply = messageTemplate(foldText(fixLink(processMarkdownLinks(telegramifyMarkdown(summaryText, 'keep')))));

        const res = await fetch(`https://api.telegram.org/bot${env.SECRET_TELEGRAM_API_TOKEN}/sendMessage`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                chat_id: groupId,
                text: formattedReply,
                parse_mode: "MarkdownV2",
            }),
        });

        if (!res.ok) {
            console.error(`[summary-job] Failed to send summary to group ${groupId}. Status: ${res.status}, Body: ${await res.text()}`);
        } else {
            console.log(`[summary-job] Successfully sent summary to group ${groupId}.`);
        }

    } catch (e: any) {
        console.error(`--- FATAL ERROR in generateAndSendSummary for group ${groupId} ---`);
        console.error("Full Error Object (Serialized):", JSON.stringify(e, null, 2));
    }
}

export default {
    async scheduled(
        controller: ScheduledController,
        env: Env,
        ctx: ExecutionContext,
    ): Promise<void> {
        console.log("[cron] Scheduled task starting:", new Date().toISOString());

        // 1. 数据库清理任务 (职责单一)
        // 在午夜执行一次，清理每个群组超过3000条的旧消息
        const cleanupTask = async () => {
            console.log("[cron-cleanup] Starting daily database cleanup.");
            await env.DB.prepare(`
                DELETE FROM Messages
                WHERE id IN (
                    SELECT id FROM (
                        SELECT id, ROW_NUMBER() OVER (PARTITION BY groupId ORDER BY timeStamp DESC) as row_num
                        FROM Messages
                    )
                    WHERE row_num > 3000
                );
            `).run();
             console.log("[cron-cleanup] Database cleanup finished.");
        };
        // 异步执行，不阻塞主流程
        ctx.waitUntil(cleanupTask());

        // 2. 获取过去24小时内的活跃群组
        const activeGroups = (await env.DB.prepare(`
            SELECT groupId
            FROM Messages
            WHERE timeStamp >= ?
            GROUP BY groupId
            HAVING COUNT(*) > 10
            ORDER BY COUNT(*) DESC;
        `).bind(Date.now() - 24 * 60 * 60 * 1000).all<{ groupId: string }>()).results ?? [];

        if (activeGroups.length === 0) {
            console.log("[cron] No active groups found to summarize. Task finished.");
            return;
        }

        console.log(`[cron] Found ${activeGroups.length} active groups.`);

        // 3. 分批处理所有活跃群组，以增加稳健性
        // 设定一个并发限制，防止瞬间向上游（AI API、Telegram API）发起过多请求，导致服务不稳定或被限流。
        const CONCURRENCY_LIMIT = 2;
        console.log(`[cron] Starting summarization with concurrency limit of ${CONCURRENCY_LIMIT}.`);

        for (let i = 0; i < activeGroups.length; i += CONCURRENCY_LIMIT) {
            const batch = activeGroups.slice(i, i + CONCURRENCY_LIMIT);
            console.log(`[cron] Processing batch ${Math.floor(i / CONCURRENCY_LIMIT) + 1} of ${Math.ceil(activeGroups.length / CONCURRENCY_LIMIT)}...`);

            const promises = batch.map(group =>
                generateAndSendSummary(group.groupId, env)
            );

            await Promise.all(promises);
        }

        console.log("[cron] All summarization tasks have been processed. Scheduled task finished.");
    },
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
		// 1. 初始化 Bot
		const bot = new Bot<MyContext>(env.SECRET_TELEGRAM_API_TOKEN);

		// 2. 注入自定义上下文
		bot.use((ctx, next) => {
			ctx.env = env;
			return next();
		});

		bot.command("status", (ctx) => ctx.reply("我家还蛮大的"));

		bot.command("query", async (ctx) => {
			const keyword = ctx.match;
			if (!keyword) {
				return ctx.reply("请输入要查询的关键词, 如 /query <keyword>");
			}
			const { results } = await ctx.env.DB.prepare(
				`SELECT id, groupId, timeStamp, userName, content, messageId, groupName FROM Messages WHERE groupId=? AND content GLOB ? ORDER BY timeStamp DESC LIMIT 50`
			).bind(ctx.chat.id, `*${keyword}*`).all();

			const responseText = `查询结果:\n${results.map((r: any) =>
				`${r.userName}: ${r.content} ${r.messageId ? `[link](${getMessageLink(r)})` : ""}`
			).join('\n')}`;

			await ctx.reply(escapeMarkdownV2(responseText), { parse_mode: "MarkdownV2" });
		});

		bot.command("ask", async (ctx) => {
			// 1. 输入验证
			const question = ctx.match;
			if (!question) {
				return ctx.reply("请输入您想问的问题，例如：\n/ask 昨天大家讨论了哪些技术话题？");
			}
			console.log(`[ask] Received question from user ${ctx.from.id}: "${question}"`);

			// 2. 提前给予用户反馈 (在群组中，作为回复)
			// 这一步是可选的，但可以提高用户体验
			const thinkingMessage = await ctx.reply("收到，我正在结合群聊上下文思考您的问题，请稍等... 🤖");

			// 3. 执行核心逻辑
			try {
				const groupId = ctx.chat.id;
				// 获取最近的消息作为上下文
				const { results } = await ctx.env.DB.prepare(`
					WITH latest_n AS (
						SELECT id, groupId, timeStamp, userName, content, messageId, groupName FROM Messages
						WHERE groupId=?
						ORDER BY timeStamp DESC
						LIMIT 1000
					)
					SELECT * FROM latest_n
					ORDER BY timeStamp ASC
				`).bind(groupId).all<R>();

				if (!results || results.length === 0) {
					// 编辑“思考中”的消息，告知用户结果
					return ctx.api.editMessageText(ctx.chat.id, thinkingMessage.message_id, "群里还没有足够多的消息让我学习，暂时无法回答。");
				}
				console.log(`[ask] Found ${results.length} messages for context.`);

				// 4. 构建结构化的 LLM 输入 (与 established best practice 一致)
				console.log("[ask] Preparing structured content for LLM...");
				const historyParts: TextPart[] = results.flatMap((r: R) => [
					{ text: `${r.userName}:` },
					{ text: r.content },
					{ text: getMessageLink(r) }
				]);

				const userContentParts: TextPart[] = [
					...historyParts,
					{ text: "---" },
					{ text: "基于以上聊天记录，请回答以下问题:" },
					{ text: question }
				];
				console.log(`[ask] Structured content prepared with ${userContentParts.length} parts.`);

				// 5. 调用 Google Generative AI API
				const result = await getGenModel(ctx.env).models.generateContent({
					model,
					contents: [
						createUserContent(userContentParts),
					],
					config: {
						temperature: 0.4,
						maxOutputTokens: 4096,
						systemInstruction: { parts: [{ text: SYSTEM_PROMPTS.answerQuestion }] },
					},
				});

				const responseText = result.text || "抱歉，我无法回答这个问题。";
				console.log("[ask] Received LLM response.");

				// 6. 格式化并发送结果 (修正点)
				const formattedReply = messageTemplate(foldText(fixLink(processMarkdownLinks(telegramifyMarkdown(responseText, 'keep')))));

				console.log(`[ask] Sending reply to chat ${ctx.chat.id}.`);

				// 使用 editMessageText 删除“思考中”的消息，并用最终答案替换它
				// 这是比发送新消息更优雅的用户体验
				await ctx.api.editMessageText(ctx.chat.id, thinkingMessage.message_id, formattedReply, {
					parse_mode: "MarkdownV2"
				});
			} catch (raw) {
				console.error(raw);

				// 2) 若是 GoogleGenerativeAIError，还可能带 response / status
				if (raw && typeof raw === "object") {
					const { name, message, stack, status, details, cause } = raw as any;
					console.error("name:", name);
					console.error("status:", status);
					console.error("details:", details);
					console.error("cause:", cause);
					console.error("stack:", stack);
				}
				// 3) 仍然给用户友好提示
				try {
					await ctx.api.editMessageText(
					ctx.chat.id,
					thinkingMessage.message_id,
					"😥 处理您的问题时发生错误，请稍后再试（后台日志已记录）。"
					);
				} catch (editErr) {
					console.error("Failed to edit thinking message:", editErr);
				}
			}
		});

		bot.command("summary", async (ctx) => {
			// 1. 输入验证
			const summaryArg = ctx.match;
			if (!summaryArg) {
				return ctx.reply("请输入要总结的时间范围或消息数量，例如：\n/summary 24h (最近24小时)\n/summary 500 (最近500条消息)");
			}
			console.log("Received summary argument:", summaryArg);

			let results: R[] = [];
			const groupId = ctx.chat.id;

			// 2. 根据参数类型获取数据
			try {
				if (summaryArg.endsWith("h")) {
					const hours = parseInt(summaryArg.slice(0, -1));
					if (isNaN(hours) || hours <= 0) throw new Error("小时数必须是正数。");

					results = (await ctx.env.DB.prepare(`
						SELECT id, groupId, timeStamp, userName, content, messageId, groupName FROM Messages
						WHERE groupId=? AND timeStamp >= ?
						ORDER BY timeStamp ASC
					`).bind(groupId, Date.now() - hours * 60 * 60 * 1000).all<R>()).results ?? [];
				} else {
					const count = parseInt(summaryArg);
					if (isNaN(count) || !Number.isFinite(count) || count <= 0) {
						throw new Error("消息数量必须是有效的正整数。");
					}

					results = (await ctx.env.DB.prepare(`
						WITH latest_n AS (
							SELECT id, groupId, timeStamp, userName, content, messageId, groupName FROM Messages
							WHERE groupId=?
							ORDER BY timeStamp DESC
							LIMIT ?
						)
						SELECT * FROM latest_n
						ORDER BY timeStamp ASC
					`).bind(groupId, Math.min(count, 4000)).all<R>()).results ?? [];
				}
			} catch (e: any) {
				return ctx.reply(`参数错误: ${e.message}`);
			}

			if (results.length === 0) {
				return ctx.reply("在指定范围内没有找到可以总结的消息。");
			}

			// 3. 执行核心逻辑
			try {
				await ctx.reply("收到，正在为您生成总结，请稍候... ✍️");
				console.log("Summarizing messages:", results.length, "messages found.");
				console.log("Message contents:", results.map(r => r.content).join("\n"));
				console.log("Preparing structured content for LLM...");

				const userContentParts: MessagePart[] = results.flatMap((r: R) => [
					`${r.userName}:`, // Part 1: 发言人
					r.content,       // Part 2: 内容
					getMessageLink(r) // Part 3: 元数据/链接
				]);

				console.log("Structured content parts prepared for LLM:", userContentParts);

				const result = await getGenModel(ctx.env).models.generateContent({
					model,
					contents: [
						createModelContent(SYSTEM_PROMPTS.summarizeChat),
						createUserContent(userContentParts),
					],
					config: {
						temperature: 0.4,
						maxOutputTokens: 4096,
					},
				});

				const messageContent = result.text || "生成总结时出现问题。";
				console.log("LLM response:", result.text);
				const formattedReply = messageTemplate(foldText(fixLink(processMarkdownLinks(telegramifyMarkdown(messageContent, 'keep')))));
				console.log("Generated summary:", formattedReply);
				// 5. 将最终结果发送回群组
				await ctx.reply(formattedReply, { parse_mode: "MarkdownV2" });

				} catch (e: any) {
					console.error("--- FATAL ERROR in /summary command ---");
					console.error("Full Error Object (Serialized):", JSON.stringify(e, null, 2));
					console.error("Error Name:", e.name);
					console.error("Error Message:", e.message);
					if (e.cause) {
						console.error("Error Cause:", e.cause);
					}
					await ctx.reply(`生成总结时发生错误，请检查后台日志获取详细信息。`);
				}
		});

		bot.on("message:text", async (ctx) => {
			if (ctx.chat.type === "private") {
				return ctx.reply("请将我添加到群组中使用。");
			}
			let content = ctx.msg.text;
			if (ctx.msg.forward_origin) {
				const fwd = ctx.msg.forward_origin.type === 'user' ? ctx.msg.forward_origin.sender_user.first_name : '未知';
				content = `转发自 ${fwd}: ${content}`;
			}
			if (ctx.msg.reply_to_message) {
				const replyToLink = getMessageLink({ groupId: ctx.chat.id.toString(), messageId: ctx.msg.reply_to_message.message_id });
				content = `回复 ${replyToLink}: ${content}`;
			}
			if (content.startsWith("http") && !content.includes(" ")) {
				content = await extractAllOGInfo(content);
			}

			await ctx.env.DB.prepare(
				`INSERT INTO Messages(id, groupId, timeStamp, userName, content, messageId, groupName) VALUES (?, ?, ?, ?, ?, ?, ?)`
			).bind(
				getMessageLink({ groupId: ctx.chat.id.toString(), messageId: ctx.msg.message_id }),
				ctx.chat.id,
				Date.now(),
				getUserName(ctx),
				content,
				ctx.msg.message_id,
				ctx.chat.title || "anonymous"
			).run();
		});

		bot.on("message:photo", async (ctx) => {
			console.log("Received a photo message:", ctx.msg.photo);
			// const photo = ctx.msg.photo?.at(-1);
			// if (!photo) return;

			// const file = await ctx.api.getFile(photo.file_id); // 等价于 ctx.getFile()

			// if (!file.file_path) {             // 理论上都会有，但以防万一
			// 	console.warn("No file_path in File response");
			// 	return;
			// }

			// const url = `https://api.telegram.org/file/bot${ctx.api.token}/${file.file_path}`;

			// const res = await fetch(url);      // 可传 AbortSignal 控制超时/取消
			// if (!res.ok) throw new Error(`Download failed: ${res.status} ${res.statusText}`);

			// const arrayBuf = await res.arrayBuffer();
			// const base64 = Buffer.from(arrayBuf).toString("base64");

			// if (!isJPEGBase64(base64).isValid) return;

			// await ctx.env.DB.prepare(`
			// 	INSERT OR REPLACE INTO Messages
			// 		(id, groupId, timeStamp, userName, content, messageId, groupName)
			// 	VALUES (?, ?, ?, ?, ?, ?, ?)
			// 	`).bind(
			// 	getMessageLink({ groupId: ctx.chat.id.toString(), messageId: ctx.msg.message_id }),
			// 	ctx.chat.id,
			// 	Date.now(),
			// 	getUserName(ctx),
			// 	"data:image/jpeg;base64," + base64,
			// 	ctx.msg.message_id,
			// 	ctx.chat.title || "anonymous"
			// 	).run();
		});

		// 处理编辑消息
		bot.on("edited_message:text", async (ctx) => {
			await ctx.env.DB.prepare(
				`INSERT OR REPLACE INTO Messages(id, groupId, timeStamp, userName, content, messageId, groupName) VALUES (?, ?, ?, ?, ?, ?, ?)`
			).bind(
				getMessageLink({ groupId: ctx.chat.id.toString(), messageId: ctx.editedMessage.message_id }),
				ctx.chat.id,
				Date.now(),
				getUserName(ctx),
				ctx.editedMessage.text,
				ctx.editedMessage.message_id,
				ctx.chat.title || "anonymous"
			).run();
		});

		// 5. 错误处理
		bot.catch((err) => {
			const ctx = err.ctx;
			console.error(`Error while handling update ${ctx.update.update_id}:`);
			const e = err.error;
			if (e instanceof GrammyError) {
				console.error("Error in request:", e.description);
			} else if (e instanceof HttpError) {
				console.error("Could not contact Telegram:", e);
			} else {
				console.error("Unknown error:", e);
			}
		});

		// 6. 启动 Webhook
		return webhookCallback(bot, "cloudflare-mod")(request);
	},
};
