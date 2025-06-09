import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import AdminMessageInput from './AdminMessageInput';
import { useAuth } from '../../context/AuthContext';
import { useTheme } from '../../context/ThemeContext';
import { IoPersonCircleOutline, IoSettingsOutline, IoPersonOutline, IoArrowBack, IoClose, IoAddCircleOutline } from 'react-icons/io5';
import { axiosInstance } from '../../api/axiosInstance';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { FaFilePdf, FaFileWord, FaFileAlt, FaFile } from 'react-icons/fa';
import { SiOpenai, SiGooglegemini } from 'react-icons/si';
import { FaRobot } from 'react-icons/fa6';
import { BiLogoMeta } from 'react-icons/bi';
import { RiOpenaiFill } from 'react-icons/ri';
import { RiSunFill, RiMoonFill } from 'react-icons/ri';

const PYTHON_URL = import.meta.env.VITE_PYTHON_API_URL || 'http://localhost:8000';

const MarkdownStyles = () => (
    <style dangerouslySetInnerHTML={{
        __html: `
        .markdown-content {
            line-height: 1.6;
            width: 100%;
        }
        
        .markdown-content h1,
        .markdown-content h2,
        .markdown-content h3 {
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        
        .markdown-content h1:first-child,
        .markdown-content h2:first-child,
        .markdown-content h3:first-child {
            margin-top: 0;
        }
        
        .markdown-content code {
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }
        
        .markdown-content pre {
            overflow-x: auto;
            border-radius: 0.375rem;
        }
        
        .markdown-content blockquote {
            font-style: italic;
            color: #6b7280;
        }
        
        .markdown-content a {
            text-decoration: none;
        }
        
        .markdown-content a:hover {
            text-decoration: underline;
        }
        
        .markdown-content table {
            border-collapse: collapse;
        }
        
        .markdown-content img {
            max-width: 100%;
            height: auto;
        }
        
        .markdown-content hr {
            border-top: 1px solid;
            margin: 1em 0;
        }
        
        .hide-scrollbar {
            -ms-overflow-style: none;
            scrollbar-width: none;
        }
        
        .hide-scrollbar::-webkit-scrollbar {
            display: none;
        }

        .progress-message {
            border-left: 3px solid #3498db;
            padding-left: 10px;
            color: #555;
            background-color: rgba(52, 152, 219, 0.05);
        }

        .progress-item {
            animation: fadeIn 0.5s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .typing-animation span {
            width: 5px;
            height: 5px;
            background-color: currentColor;
            border-radius: 50%;
            display: inline-block;
            margin: 0 1px;
            animation: typing 1.3s infinite ease-in-out;
        }

        .typing-animation span:nth-child(1) {
            animation-delay: 0s;
        }

        .typing-animation span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-animation span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-5px); }
        }
    `}} />
);

const modelIcons = {
    'gpt-4': <RiOpenaiFill className="text-green-500" size={18} />,
    'gpt-4o-mini': <SiOpenai className="text-green-400" size={16} />,
    'claude': <FaRobot className="text-purple-400" size={16} />,
    'gemini': <SiGooglegemini className="text-blue-400" size={16} />,
    'llama': <BiLogoMeta className="text-blue-500" size={18} />
};

const getDisplayModelName = (modelType) => {
    if (modelType === 'openrouter/auto') return 'router-engine';
    return modelType;
};

const AdminChat = () => {
    const { gptId } = useParams();
    const navigate = useNavigate();
    const location = useLocation();
    const { user, loading: authLoading } = useAuth();
    const { isDarkMode, toggleTheme } = useTheme();
    const [isProfileOpen, setIsProfileOpen] = useState(false);
    const [userData, setUserData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isFetchingGpt, setIsFetchingGpt] = useState(false);
    const [gptData, setGptData] = useState(null);
    const [messages, setMessages] = useState([]);
    const [collectionName, setCollectionName] = useState(null);
    const messagesEndRef = useRef(null);
    const [userDocuments, setUserDocuments] = useState([]);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [streamingMessage, setStreamingMessage] = useState(null);
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [backendAvailable, setBackendAvailable] = useState(null);
    const [hasInteracted, setHasInteracted] = useState(false);
    const [conversationMemory, setConversationMemory] = useState([]);
    const [hasNotifiedGptOpened, setHasNotifiedGptOpened] = useState(false);
    const [conversationId, setConversationId] = useState(null);
    const [isInitialLoading, setIsInitialLoading] = useState(false);
    const [currentConversationId, setCurrentConversationId] = useState(null);
    const [loading, setLoading] = useState({ message: false });
    const [webSearchEnabled, setWebSearchEnabled] = useState(false);
    const [apiKeys, setApiKeys] = useState({});

    // Use effect to handle user data changes
    useEffect(() => {
        if (user) {
            setUserData(user);
        }
    }, [user]);

    // Create a better fetchApiKeys function with retry logic
    const fetchApiKeysFromBackend = async (retry = 3) => {
        try {
            if (!user?._id) {
                console.warn("Cannot fetch API keys - user not authenticated yet");
                return {};
            }
            
            console.log("Fetching API keys from backend...");
            const response = await axiosInstance.get('/api/auth/user/api-keys', {
                withCredentials: true
            });
            
            if (response.data && response.data.success) {
                const keys = response.data.apiKeys || {};
                console.log("API keys fetched successfully:", Object.keys(keys));
                // Update state with the fetched keys
                setApiKeys(keys);
                return keys;
            }
            return {};
        } catch (error) {
            console.error("Failed to fetch API keys from server:", error);
            if (retry > 0) {
                console.log(`Retrying API key fetch (${retry} attempts left)...`);
                await new Promise(resolve => setTimeout(resolve, 1000));
                return fetchApiKeysFromBackend(retry - 1);
            }
            return {};
        }
    };

    // Add a useEffect to fetch API keys after user is authenticated
    useEffect(() => {
        if (user?._id) {
            fetchApiKeysFromBackend().then(keys => {
                console.log("Initial API keys fetched:", Object.keys(keys));
            });
        }
    }, [user]);

    // Update notifyGptOpened function
    const notifyGptOpened = async (gptData, userData) => {
        try {
            if (!gptData || !userData || !gptData._id || hasNotifiedGptOpened) {
                return;
            }

            const fileUrls = gptData.knowledgeFiles?.map(file => file.fileUrl).filter(url =>
                url && (url.startsWith('http://') || url.startsWith('https://'))
            ) || [];
            
            // Get API keys from backend instead of localStorage
            const apiKeys = await fetchApiKeysFromBackend();
            console.log("Fetched API keys from backend:", Object.keys(apiKeys));

            const response = await axios.post(
                `${PYTHON_URL}/gpt-opened`,
                {
                    user_email: userData.email,
                    gpt_name: gptData.name,
                    gpt_id: gptData._id,
                    file_urls: fileUrls,
                    schema: {
                        model: gptData.model,
                        instructions: gptData.instructions,
                        capabilities: gptData.capabilities
                    },
                    api_keys: apiKeys
                },
                {
                    headers: {
                        'Content-Type': 'application/json',
                    }
                }
            );

            if (response.data.success) {
                setCollectionName(response.data.collection_name);
                setHasNotifiedGptOpened(true);
            }
        } catch (error) {
            console.error("Error notifying GPT opened:", error);
        }
    };

    // Get conversationId from URL query params
    useEffect(() => {
        const params = new URLSearchParams(location.search);
        const convId = params.get('conversationId');
        setCurrentConversationId(convId);
    }, [location.search]);

    // Main useEffect for fetching data
    useEffect(() => {
        if (!gptId) {
            setGptData(null);
            setMessages([]);
            setConversationMemory([]);
            setIsInitialLoading(false);
            return;
        }
        if (authLoading) {
            return;
        }
        if (!authLoading && !user) {
            console.warn("AdminChat: Auth finished, no user.");
            setIsInitialLoading(false);
            setGptData({ _id: gptId, name: "Admin Chat", description: "Admin user required.", model: "gpt-4o-mini" });
            setMessages([]);
            setConversationMemory([]);
            return;
        }
        if (user.role !== 'admin') {
            console.warn("AdminChat: Non-admin user trying to access.");
            setIsInitialLoading(false);
            navigate('/user/collections');
            return;
        }

        setIsInitialLoading(true);

        const fetchAdminChatData = async () => {
            let fetchedGptData = null;
            let conversationMessages = [];
            let conversationMemorySlice = [];

            try {
                const gptResponse = await axiosInstance.get(`/api/custom-gpts/${gptId}`, { withCredentials: true });
                if (gptResponse.data?.success && gptResponse.data.customGpt) {
                    fetchedGptData = gptResponse.data.customGpt;
                    setGptData(fetchedGptData);
                    const sanitizedEmail = (user.email || 'admin').replace(/[^a-zA-Z0-9]/g, '_');
                    const sanitizedGptName = (fetchedGptData.name || 'gpt').replace(/[^a-zA-Z0-9]/g, '_');
                    setCollectionName(`kb_${sanitizedEmail}_${sanitizedGptName}_${gptId}`);
                    notifyGptOpened(fetchedGptData, user).catch(err => console.warn("[AdminChat] Notify error:", err));
                } else {
                    console.warn("[AdminChat] Failed GPT fetch:", gptResponse.data);
                    fetchedGptData = { _id: gptId, name: "Assistant", description: "Details unavailable.", model: "gpt-4o-mini" };
                    setGptData(fetchedGptData);
                }

                if (currentConversationId) {
                    const historyResponse = await axiosInstance.get(`/api/chat-history/admin/conversation/${currentConversationId}`, { withCredentials: true });
                    if (historyResponse.data?.success && historyResponse.data.conversation?.messages?.length > 0) {
                        const { conversation } = historyResponse.data;
                        conversationMessages = conversation.messages.map((msg, index) => ({
                            id: `${conversation._id}-${index}-${msg.timestamp || Date.now()}`,
                            role: msg.role,
                            content: msg.content,
                            timestamp: new Date(msg.timestamp || conversation.createdAt)
                        }));
                        conversationMemorySlice = conversation.messages.slice(-10).map(msg => ({
                            role: msg.role,
                            content: msg.content,
                            timestamp: msg.timestamp || conversation.createdAt
                        }));
                    } else {
                        conversationMessages = [{
                            id: Date.now(),
                            role: 'system',
                            content: `Could not load conversation ${currentConversationId}. It might be empty or not found.`,
                            timestamp: new Date()
                        }];
                        conversationMemorySlice = [];
                    }
                } else {
                    conversationMessages = [];
                    conversationMemorySlice = [];
                }

                setMessages(conversationMessages);
                setConversationMemory(conversationMemorySlice);
            } catch (err) {
                console.error("[AdminChat] Error during fetch:", err);
                setGptData(fetchedGptData || { _id: gptId, name: "Assistant", description: "Error loading data.", model: "gpt-4o-mini" });
                setMessages([{ id: Date.now(), role: 'system', content: `Error loading chat data: ${err.message}`, timestamp: new Date() }]);
                setConversationMemory([]);
            } finally {
                setIsInitialLoading(false);
            }
        };

        fetchAdminChatData();

        return () => {
            setIsInitialLoading(false);
            setLoading(prev => ({ ...prev, message: false }));
        };
    }, [gptId, user, authLoading, currentConversationId]);

    const handlePromptClick = (item) => {
        handleChatSubmit(item.prompt);
    };

    const saveMessageToHistory = async (message, role) => {
        try {
            if (!user?._id || !gptData || !message || !message.trim()) {
                console.warn('Cannot save message - missing data:', {
                    userId: user?._id,
                    gptId: gptData?._id,
                    hasMessage: !!message,
                    role
                });
                return null;
            }

            const payload = {
                userId: user._id,
                gptId: gptData._id,
                gptName: gptData.name || 'AI Assistant',
                message: message.trim(),
                role: role,
                model: gptData.model || 'gpt-4o-mini'
            };

            if (conversationId) {
                payload.conversationId = conversationId;
            }

            const response = await axiosInstance.post('/api/chat-history/save', payload, {
                withCredentials: true
            });

            if (response.data && response.data.conversation && response.data.conversation._id) {
                setConversationId(response.data.conversation._id);
            }

            return response.data;
        } catch (error) {
            console.error(`Error saving ${role} message to history:`, error.response?.data || error.message);
            return null;
        }
    };

    // Update handleChatSubmit function
    const handleChatSubmit = async (message) => {
        if (!message.trim()) return;

        try {
            const isNewChat = messages.length === 0 && !conversationId;
            
            // Include files in the user message
            const userMessage = {
                id: Date.now(),
                role: 'user',
                content: message,
                timestamp: new Date(),
                files: uploadedFiles.length > 0 ? [...uploadedFiles] : []
            };

            await saveMessageToHistory(message, 'user');
            setMessages(prev => [...prev, userMessage]);
            
            // Save current files for this message then clear them for next message
            const currentFiles = [...uploadedFiles];
            if (uploadedFiles.length > 0) {
                setUploadedFiles([]); // Clear files after using them
            }

            // Important: Clear any existing streaming message first
            setStreamingMessage(null);
            
            // Then set loading state
            setLoading(prev => ({ ...prev, message: true }));
            setHasInteracted(true);

            const updatedMemory = [...conversationMemory];
            if (updatedMemory.length >= 10) {
                updatedMemory.splice(0, updatedMemory.length - 9);
            }
            updatedMemory.push({
                role: 'user',
                content: message,
                timestamp: new Date().toISOString()
            });
            setConversationMemory(updatedMemory);

            // Get API keys from backend
            const apiKeys = await fetchApiKeysFromBackend();
            console.log("Using API keys for chat:", Object.keys(apiKeys));

            const payload = {
                message,
                gpt_id: gptId,
                user_email: user?.email || 'unknown_admin',
                gpt_name: gptData?.name || 'unknown_gpt',
                user_documents: userDocuments,
                model: gptData?.model || 'openrouter/auto',
                memory: updatedMemory,
                history: messages.slice(-6).map(msg => ({
                    role: msg.role,
                    content: msg.content
                })),
                system_prompt: gptData?.instructions || null,
                web_search_enabled: gptData?.capabilities?.webBrowsing || false,
                api_keys: apiKeys,
                is_new_chat: isNewChat
            };

            if (!payload.user_email) {
                payload.user_email = user?.email || 'admin@system.com';
            }
            if (!payload.gpt_name) {
                payload.gpt_name = gptData?.name || 'Admin Chat';
            }
            if (!payload.gpt_id && gptData?._id) {
                payload.gpt_id = gptData._id;
            } else if (!payload.gpt_id && gptId) {
                payload.gpt_id = gptId;
            }

            if (!payload.gpt_id) {
                throw new Error("GPT ID is missing, cannot send message.");
            }

            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 30000);

                const response = await fetch(`${PYTHON_URL}/chat-stream`, {
                    method: "POST",
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    credentials: 'include',
                    signal: controller.signal,
                    body: JSON.stringify(payload)
                });

                clearTimeout(timeoutId);

                if (response.ok) {
                    await handleStreamingResponse(response);
                } else {
                    console.error("Stream response not OK:", response.status, response.statusText);
                    const errorText = await response.text();
                    console.error("Stream error response body:", errorText);
                    throw new Error(`HTTP error! status: ${response.status} - ${errorText || response.statusText}`);
                }
            } catch (streamingError) {
                console.warn("Streaming failed, falling back to regular chat API:", streamingError);

                const fallbackResponse = await axios.post(
                    `${PYTHON_URL}/chat`,
                    payload,
                    {
                        headers: {
                            'Content-Type': 'application/json',
                        },
                    }
                );

                if (fallbackResponse.data && fallbackResponse.data.success && fallbackResponse.data.answer) {
                    const aiResponse = {
                        id: Date.now() + 1,
                        role: 'assistant',
                        content: fallbackResponse.data.answer,
                        timestamp: new Date()
                    };

                    setMessages(prev => [...prev, aiResponse]);
                    await saveMessageToHistory(aiResponse.content, 'assistant');
                } else {
                    const errorContent = fallbackResponse.data?.answer || "Failed to get response from fallback API.";
                    const errorResponse = {
                        id: Date.now() + 1,
                        role: 'assistant',
                        content: errorContent,
                        timestamp: new Date()
                    };
                    setMessages(prev => [...prev, errorResponse]);
                    await saveMessageToHistory(errorContent, 'assistant');
                }
            }
        } catch (err) {
            console.error("Error in handleChatSubmit:", err);
            const errorContent = `I'm sorry, I couldn't process your request: ${err.message}`;
            const errorResponse = {
                id: Date.now() + 1,
                role: 'assistant',
                content: errorContent,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, errorResponse]);
            await saveMessageToHistory(errorContent, 'assistant');
            setStreamingMessage(null);
        } finally {
            setLoading(prev => ({ ...prev, message: false }));
        }
    };

    const toggleProfile = () => {
        setIsProfileOpen(!isProfileOpen);
    };

    const handleGoBack = () => {
        navigate(-1);
    };

    const mockUser = {
        name: "Admin User",
        email: "admin@example.com",
        profilePic: null
    };

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Update handleFileUpload function
    const handleFileUpload = async (files) => {
        if (!files.length || !gptData) return;

        try {
            setIsUploading(true);
            setUploadProgress(0);
            
            // Show files immediately to reduce perceived latency
            const fileObjects = Array.from(files).map(file => ({
                name: file.name,
                size: file.size,
                type: file.type
            }));
            setUploadedFiles(fileObjects);

            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }
            formData.append('user_email', userData?.email || 'user@example.com');
            formData.append('gpt_id', gptData._id);
            formData.append('gpt_name', gptData.name);
            formData.append('collection_name', collectionName || gptData._id);
            formData.append('is_user_document', 'true');
            formData.append('system_prompt', gptData?.instructions || '');

            // Simulate faster initial progress (psychological trick to reduce perceived latency)
            setUploadProgress(15); // Jump to 15% immediately
            setTimeout(() => setUploadProgress(30), 100); // 30% after 100ms
            
            // Get API keys from backend
            const apiKeys = await fetchApiKeysFromBackend();
            formData.append('api_keys', JSON.stringify(apiKeys));

            // Use a higher timeout and buffer to ensure large file uploads complete
            const response = await axios.post(
                `${PYTHON_URL}/upload-chat-files`,
                formData,
                {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                    timeout: 60000, // 60 seconds timeout
                    onUploadProgress: (progressEvent) => {
                        const percentCompleted = Math.round(
                            (progressEvent.loaded * 60) / (progressEvent.total || 100)
                        );
                        setUploadProgress(30 + Math.min(percentCompleted, 60)); // Start from 30% to 90%
                    }
                }
            );

            setUploadProgress(100);
            setTimeout(() => setIsUploading(false), 200); // Shorter delay before hiding

            if (response.data.success) {
                setUserDocuments(response.data.file_urls || []);
            } else {
                throw new Error(response.data.message || "Failed to process files");
            }
        } catch (error) {
            console.error("Error uploading files:", error);
            setIsUploading(false);
            // Show error to user
            setMessages(prev => [...prev, {
                id: Date.now(),
                role: 'system',
                content: `Error uploading files: ${error.message}`,
                timestamp: new Date()
            }]);
        }
    };

    const getFileIcon = (filename) => {
        if (!filename) return <FaFile size={14} />;

        const extension = filename.split('.').pop().toLowerCase();
        switch (extension) {
            case 'pdf':
                return <FaFilePdf size={14} className="text-red-400 dark:text-red-300" />;
            case 'doc':
            case 'docx':
                return <FaFileWord size={14} className="text-blue-400 dark:text-blue-300" />;
            case 'txt':
                return <FaFileAlt size={14} />;
            default:
                return <FaFile size={14} />;
        }
    };

    const handleRemoveUploadedFile = (indexToRemove) => {
        setUploadedFiles(prevFiles => prevFiles.filter((_, index) => index !== indexToRemove));
    };

    useEffect(() => {
        const checkBackendAvailability = async () => {
            try {
                await axios.get(`${PYTHON_URL}/gpt-collection-info/test/test`);
                setBackendAvailable(true);
            } catch (error) {
                if (error.code === "ERR_NETWORK") {
                    console.error("Backend server appears to be offline:", error);
                    setBackendAvailable(false);
                } else {
                    setBackendAvailable(true);
                }
            }
        };

        checkBackendAvailability();
    }, []);

    const handleStreamingResponse = async (response) => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let doneStreaming = false;
        let sourcesInfo = null;
        let streamError = null;
        let progressMessages = "";
        let hasReceivedContent = false;

        const messageId = streamingMessage?.id || Date.now();

        try {
            // Create initial progress message
            setStreamingMessage({
                id: messageId,
                role: 'assistant',
                content: "🔍 Searching for information...",
                isStreaming: true,
                isProgress: true,
                timestamp: new Date()
            });

            while (!doneStreaming) {
                const { done, value } = await reader.read();

                if (done) {
                    doneStreaming = true;
                    break;
                }

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n\n').filter(line => line.trim().startsWith('data: '));

                for (const line of lines) {
                    try {
                        const jsonStr = line.substring(6);
                        const parsed = JSON.parse(jsonStr);

                        if (parsed.type === 'error' || parsed.error) {
                            streamError = parsed.error || parsed.detail || 'Unknown streaming error';
                            console.error(`[Stream ${messageId}] Streaming Error:`, streamError);
                            buffer = `Error: ${streamError}`;
                            doneStreaming = true;
                            setStreamingMessage(prev =>
                                prev ? { ...prev, content: buffer, isStreaming: false, isError: true } :
                                    { id: messageId, role: 'assistant', content: buffer, isStreaming: false, isError: true, timestamp: new Date() }
                            );
                            break;
                        }

                        if (parsed.type === 'done') {
                            doneStreaming = true;
                            break;
                        }

                        if (parsed.type === 'content') {
                            hasReceivedContent = true;
                            // If we were showing progress and now getting content, start fresh with content
                            if (progressMessages && !buffer) {
                                buffer = parsed.data;
                                setStreamingMessage({
                                    id: messageId,
                                    role: 'assistant',
                                    content: buffer,
                                    isStreaming: true,
                                    isProgress: false,
                                    timestamp: new Date()
                                });
                            } else {
                                buffer += parsed.data;
                                setStreamingMessage(prev =>
                                    prev ? { 
                                        ...prev, 
                                        content: buffer, 
                                        isStreaming: true, 
                                        isProgress: false, 
                                        isError: false 
                                    } : { 
                                        id: messageId, 
                                        role: 'assistant', 
                                        content: buffer, 
                                        isStreaming: true, 
                                        isProgress: false, 
                                        isError: false, 
                                        timestamp: new Date() 
                                    }
                                );
                            }
                        }

                        // Show progress updates
                        if (parsed.type === 'progress') {
                            progressMessages += progressMessages ? `\n• ${parsed.data}` : `🔍 Searching for information...\n• ${parsed.data}`;
                            // Only update with progress if we haven't received actual content yet
                            if (!hasReceivedContent) {
                                setStreamingMessage(prev => ({
                                    ...prev,
                                    content: progressMessages,
                                    isStreaming: true,
                                    isProgress: true
                                }));
                            }
                        }

                        if (parsed.type === 'sources_info') {
                            sourcesInfo = parsed.data;
                            // Only update with progress if we haven't received actual content yet
                            if (!hasReceivedContent) {
                                progressMessages += `\n\n[Sources: ${sourcesInfo.documents_retrieved_count} documents, ${sourcesInfo.retrieval_time_ms}ms]`;
                                setStreamingMessage(prev => ({
                                    ...prev,
                                    content: progressMessages,
                                    isStreaming: true,
                                    isProgress: true
                                }));
                            }
                        }
                    } catch (e) {
                        console.error(`[Stream ${messageId}] Error parsing line:`, e, "Line:", line);
                    }
                }
            }

            // Only show the "no response" message if we truly have no content
            if (!buffer && !hasReceivedContent && !streamError) {
                console.warn(`[Stream ${messageId}] Stream ended with no content.`);
                buffer = "No response generated. Please try rephrasing your query or check the uploaded documents.";
                streamError = true;
            }

            // Ensure we have content in the buffer - use progress messages if that's all we have
            if (!buffer && progressMessages) {
                buffer = `I searched your documents but couldn't generate a good response. Here's what I found:\n\n${progressMessages}`;
            }

            setStreamingMessage(prev =>
                prev ? {
                    ...prev,
                    content: buffer || prev.content,
                    isStreaming: false,
                    isLoading: false,
                    isProgress: false,
                    isError: !!streamError
                } : {
                    id: messageId,
                    role: 'assistant',
                    content: buffer,
                    isStreaming: false,
                    isLoading: false,
                    isProgress: false,
                    isError: !!streamError,
                    timestamp: new Date()
                }
            );

            await saveMessageToHistory(buffer, 'assistant');
        } catch (err) {
            console.error(`[Stream ${messageId}] Error reading stream:`, err);
            buffer = `Error reading response stream: ${err.message}`;
            setStreamingMessage(prev =>
                prev ? { ...prev, content: buffer, isStreaming: false, isLoading: false, isError: true } :
                    { id: messageId, role: 'assistant', content: buffer, isStreaming: false, isLoading: false, isError: true, timestamp: new Date() }
            );
            await saveMessageToHistory(buffer, 'assistant');
        } finally {
            setLoading(prev => ({ ...prev, message: false }));
        }
    };

    useEffect(() => {
        if (streamingMessage && !streamingMessage.isStreaming) {
            setMessages(prev => {
                const exists = prev.some(m =>
                    m.content === streamingMessage.content &&
                    m.timestamp.getTime() === streamingMessage.timestamp.getTime()
                );
                if (exists) return prev;
                return [...prev, { ...streamingMessage }];
            });

            setConversationMemory(prev => [...prev, {
                role: 'assistant',
                content: streamingMessage.content,
                timestamp: new Date().toISOString()
            }]);

            setTimeout(() => {
                setStreamingMessage(null);
                setLoading(prev => ({ ...prev, message: false }));
            }, 100);
        }
    }, [streamingMessage]);

    // Determine if the web search toggle should be shown
    const showWebSearchToggle = gptData?.capabilities?.webBrowsing || false;

    const handleNewChat = () => {
        setMessages([]);
        setConversationMemory([]);
        setHasInteracted(false);
        setUserDocuments([]);
        setUploadedFiles([]);
        setConversationId(null); // Reset conversation ID
        
        // Also tell the backend to clear memory for this session
        if (gptData && userData) {
            const sessionId = `user_${userData.email.replace('@', '_').replace('.', '_')}_gpt_${gptData._id}`;
            // No need to make a separate request, the next query will include is_new_chat=true
            console.log(`Starting new chat for session ${sessionId}`);
        }
    };

    return (
        <>
            <MarkdownStyles />
            <div className='flex flex-col h-screen bg-white dark:bg-black text-black dark:text-white overflow-hidden'>
                <div className="flex-shrink-0 bg-white dark:bg-black px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                        {gptId && (
                            <button
                                onClick={handleGoBack}
                                className="text-gray-500 dark:text-gray-400 hover:text-black dark:hover:text-white p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center justify-center w-10 h-10"
                                aria-label="Go back"
                            >
                                <IoArrowBack size={20} />
                            </button>
                        )}
                        
                        <button
                            onClick={handleNewChat}
                            className="text-gray-500 dark:text-gray-400 hover:text-black dark:hover:text-white p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center justify-center w-10 h-10"
                            aria-label="New Chat"
                        >
                            <IoAddCircleOutline size={24} /> 
                        </button>

                        {gptData && (
                            <div className="ml-2 text-sm md:text-base font-medium flex items-center">
                                <span className="mr-1">{gptData.name}</span>
                                {gptData.model && (
                                    <div className="flex items-center ml-2 text-xs md:text-sm px-2 py-0.5 bg-gray-100 dark:bg-gray-800 rounded-full">
                                        {modelIcons[gptData.model === 'openrouter/auto' ? 'router-engine' : gptData.model] || null}
                                        <span>{gptData.model === 'openrouter/auto' ? 'router-engine' : (gptData.model === 'gpt-4o-mini' ? 'GPT-4o Mini' : gptData.model)}</span>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={toggleTheme}
                            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                            title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
                        >
                            {isDarkMode ? 
                                <RiSunFill size={20} className="text-yellow-400" /> : 
                                <RiMoonFill size={20} className="text-gray-700" />
                            }
                        </button>
                        <div className="relative">
                            <button
                                onClick={toggleProfile}
                                className="w-10 h-10 rounded-full overflow-hidden border-2 border-gray-300 dark:border-white/20 hover:border-blue-500 dark:hover:border-white/40 transition-colors"
                            >
                                {(userData || mockUser)?.profilePic ? (
                                    <img
                                        src={(userData || mockUser).profilePic}
                                        alt="Profile"
                                        className="w-full h-full object-cover"
                                    />
                                ) : (
                                    <div className="w-full h-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                                        <IoPersonCircleOutline size={24} className="text-gray-500 dark:text-white" />
                                    </div>
                                )}
                            </button>

                            {isProfileOpen && (
                                <div className="absolute top-12 right-0 w-64 bg-white dark:bg-[#1e1e1e] rounded-xl shadow-lg border border-gray-200 dark:border-white/10 overflow-hidden z-30">
                                    <div className="p-4 border-b border-gray-200 dark:border-white/10">
                                        <p className="font-medium text-gray-900 dark:text-white">
                                            {userData?.name || mockUser.name}
                                        </p>
                                        <p className="text-sm text-gray-500 dark:text-gray-400 truncate">
                                            {userData?.email || mockUser.email}
                                        </p>
                                    </div>
                                    <div className="py-1">
                                        <button className="w-full px-4 py-2.5 text-left flex items-center space-x-3 hover:bg-gray-100 dark:hover:bg-white/5 text-gray-700 dark:text-gray-300">
                                            <IoPersonOutline size={18} />
                                            <span>Profile</span>
                                        </button>
                                        <button className="w-full px-4 py-2.5 text-left flex items-center space-x-3 hover:bg-gray-100 dark:hover:bg-white/5 text-gray-700 dark:text-gray-300" onClick={() => navigate('/admin/settings')}>
                                            <IoSettingsOutline size={18} />
                                            <span>Settings</span>
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-4 flex flex-col bg-white dark:bg-black hide-scrollbar">
                    <div className="w-full max-w-3xl mx-auto flex-1 flex flex-col space-y-4 pb-4">
                        {isInitialLoading ? (
                            <div className="flex-1 flex flex-col items-center justify-center p-20">
                                <span className="mt-4 text-sm">Loading chat...</span>
                            </div>
                        ) : isFetchingGpt ? (
                            <div className="flex-1 flex items-center justify-center">
                                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 dark:border-blue-400"></div>
                            </div>
                        ) : messages.length === 0 ? (
                            <div className="flex-1 flex flex-col items-center justify-center text-center px-2">
                                {gptId && gptData ? (
                                    <>
                                        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-600 to-purple-600 flex items-center justify-center mb-4">
                                            {gptData.imageUrl ? (
                                                <img src={gptData.imageUrl} alt={gptData.name} className="w-full h-full object-cover rounded-full" />
                                            ) : (
                                                <span className="text-2xl text-white">{gptData.name?.charAt(0) || '?'}</span>
                                            )}
                                        </div>
                                        <h2 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">{gptData.name}</h2>
                                        <p className="text-gray-500 dark:text-gray-400 max-w-md">{gptData.description || 'Start a conversation...'}</p>
                                        {gptData.conversationStarter && (
                                            <div
                                                onClick={() => handleChatSubmit(gptData.conversationStarter)}
                                                className="mt-5 max-w-xs p-3 bg-gray-100 dark:bg-gray-800/70 border border-gray-300 dark:border-gray-700/70 rounded-lg text-left cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 hover:border-gray-400 dark:hover:border-gray-600/70 transition-colors"
                                            >
                                                <p className="text-gray-800 dark:text-white text-sm">
                                                    {gptData.conversationStarter.length > 40
                                                        ? gptData.conversationStarter.substring(0, 40) + '...'
                                                        : gptData.conversationStarter
                                                    }
                                                </p>
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <>
                                        <h1 className='text-2xl sm:text-3xl md:text-4xl font-bold mb-2 text-gray-900 dark:text-white'>Welcome to AI Agent</h1>
                                        <span className='text-base sm:text-lg md:text-xl font-medium text-gray-500 dark:text-gray-400 mb-8 block'>How can I assist you today?</span>
                                    </>
                                )}
                            </div>
                        ) : (
                            <>
                                {messages.map(message => (
                                    <div
                                        key={message.id}
                                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                    >
                                        <div
                                            className={`${message.role === 'user'
                                                    ? 'bg-black/10 dark:bg-white/80 text-black font-[16px] dark:text-black rounded-br-none max-w-max '
                                                    : 'assistant-message text-black font-[16px] dark:text-white rounded-bl-none w-full max-w-3xl'
                                                } rounded-2xl px-4 py-2`}
                                        >
                                            {message.role === 'user' ? (
                                                <>
                                                    <p className="whitespace-pre-wrap">{message.content}</p>
                                                    
                                                    {/* Display files attached to this message */}
                                                    {message.files && message.files.length > 0 && (
                                                        <div className="mt-2 flex flex-wrap gap-2">
                                                            {message.files.map((file, index) => (
                                                                <div
                                                                    key={`${file.name}-${index}`}
                                                                    className="flex items-center py-1 px-2 bg-gray-50 dark:bg-gray-800/50 rounded-md border border-gray-200 dark:border-gray-700/50 max-w-fit"
                                                                >
                                                                    <div className="mr-1.5 text-gray-500 dark:text-gray-400">
                                                                        {getFileIcon(file.name)}
                                                                    </div>
                                                                    <span className="text-xs font-medium text-gray-700 dark:text-gray-300 truncate max-w-[140px]">
                                                                        {file.name}
                                                                    </span>
                                                                    {file.size && (
                                                                        <div className="text-[10px] text-gray-500 ml-1 whitespace-nowrap">
                                                                            {Math.round(file.size / 1024)} KB
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    )}
                                                </>
                                            ) : (
                                                <div className="markdown-content">
                                                    <ReactMarkdown
                                                        remarkPlugins={[remarkGfm]}
                                                        rehypePlugins={[rehypeRaw]}
                                                        components={{
                                                            h1: ({ node, ...props }) => <h1 className="text-xl font-bold my-3" {...props} />,
                                                            h2: ({ node, ...props }) => <h2 className="text-lg font-bold my-2" {...props} />,
                                                            h3: ({ node, ...props }) => <h3 className="text-md font-bold my-2" {...props} />,
                                                            h4: ({ node, ...props }) => <h4 className="font-bold my-2" {...props} />,
                                                            p: ({ node, ...props }) => <p className="my-2" {...props} />,
                                                            ul: ({ node, ...props }) => <ul className="list-disc pl-5 my-2" {...props} />,
                                                            ol: ({ node, ...props }) => <ol className="list-decimal pl-5 my-2" {...props} />,
                                                            li: ({ node, index, ...props }) => <li key={index} className="my-1" {...props} />,
                                                            a: ({ node, ...props }) => <a className="text-blue-400 hover:underline" {...props} />,
                                                            blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-gray-500 dark:border-gray-400 pl-4 my-3 italic" {...props} />,
                                                            code({ node, inline, className, children, ...props }) {
                                                                const match = /language-(\w+)/.exec(className || '');
                                                                return !inline && match ? (
                                                                    <SyntaxHighlighter
                                                                        style={atomDark}
                                                                        language={match[1]}
                                                                        PreTag="div"
                                                                        className="rounded-md my-3"
                                                                        {...props}
                                                                    >
                                                                        {String(children).replace(/\n$/, '')}
                                                                    </SyntaxHighlighter>
                                                                ) : (
                                                                    <code className={`${inline ? 'bg-gray-300 dark:bg-gray-600 px-1 py-0.5 rounded text-sm' : ''} ${className}`} {...props}>
                                                                        {children}
                                                                    </code>
                                                                );
                                                            },
                                                            table: ({ node, ...props }) => (
                                                                <div className="overflow-x-auto my-3">
                                                                    <table className="min-w-full border border-gray-400 dark:border-gray-500" {...props} />
                                                                </div>
                                                            ),
                                                            thead: ({ node, ...props }) => <thead className="bg-gray-300 dark:bg-gray-600" {...props} />,
                                                            tbody: ({ node, ...props }) => <tbody className="divide-y divide-gray-400 dark:divide-gray-500" {...props} />,
                                                            tr: ({ node, ...props }) => <tr className="hover:bg-gray-300 dark:hover:bg-gray-600" {...props} />,
                                                            th: ({ node, ...props }) => <th className="px-4 py-2 text-left font-medium" {...props} />,
                                                            td: ({ node, ...props }) => <td className="px-4 py-2" {...props} />,
                                                        }}
                                                    >
                                                        {message.content}
                                                    </ReactMarkdown>
                                                </div>
                                            )}
                                            <div className={`text-xs mt-2 text-right ${message.role === 'user' ? 'text-blue-50/80' : 'text-gray-400/80'}`}>
                                            </div>
                                        </div>
                                    </div>
                                ))}

                                {streamingMessage ? (
                                    <div className="flex justify-start">
                                        <div className={`w-full max-w-3xl rounded-2xl px-4 py-2 assistant-message text-black dark:text-white rounded-bl-none ${streamingMessage.isProgress ? 'progress-message' : ''}`}>
                                            <div className="markdown-content">
                                                {streamingMessage.isProgress ? (
                                                    // Progress message with animation
                                                    <div>
                                                        {streamingMessage.content.split('\n').map((line, i) => (
                                                            <div key={i} className={`progress-item ${i > 0 ? 'mt-1' : ''}`} style={{animationDelay: `${i * 0.1}s`}}>
                                                                {line}
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    // Regular markdown content
                                                    <ReactMarkdown
                                                        remarkPlugins={[remarkGfm]}
                                                        rehypePlugins={[rehypeRaw]}
                                                        components={{
                                                            h1: ({ node, ...props }) => <h1 className="text-xl font-bold my-3" {...props} />,
                                                            h2: ({ node, ...props }) => <h2 className="text-lg font-bold my-2" {...props} />,
                                                            h3: ({ node, ...props }) => <h3 className="text-md font-bold my-2" {...props} />,
                                                            h4: ({ node, ...props }) => <h4 className="font-bold my-2" {...props} />,
                                                            p: ({ node, ...props }) => <p className="my-2" {...props} />,
                                                            ul: ({ node, ...props }) => <ul className="list-disc pl-5 my-2" {...props} />,
                                                            ol: ({ node, ...props }) => <ol className="list-decimal pl-5 my-2" {...props} />,
                                                            li: ({ node, index, ...props }) => <li key={index} className="my-1" {...props} />,
                                                            a: ({ node, ...props }) => <a className="text-blue-400 hover:underline" {...props} />,
                                                            blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-gray-500 dark:border-gray-400 pl-4 my-3 italic" {...props} />,
                                                            code({ node, inline, className, children, ...props }) {
                                                                const match = /language-(\w+)/.exec(className || '');
                                                                return !inline && match ? (
                                                                    <SyntaxHighlighter
                                                                        style={atomDark}
                                                                        language={match[1]}
                                                                        PreTag="div"
                                                                        className="rounded-md my-3"
                                                                        {...props}
                                                                    >
                                                                        {String(children).replace(/\n$/, '')}
                                                                    </SyntaxHighlighter>
                                                                ) : (
                                                                    <code className={`${inline ? 'bg-gray-300 dark:bg-gray-600 px-1 py-0.5 rounded text-sm' : ''} ${className}`} {...props}>
                                                                        {children}
                                                                    </code>
                                                                );
                                                            },
                                                            table: ({ node, ...props }) => (
                                                                <div className="overflow-x-auto my-3">
                                                                    <table className="min-w-full border border-gray-400 dark:border-gray-500" {...props} />
                                                                </div>
                                                            ),
                                                            thead: ({ node, ...props }) => <thead className="bg-gray-300 dark:bg-gray-600" {...props} />,
                                                            tbody: ({ node, ...props }) => <tbody className="divide-y divide-gray-400 dark:divide-gray-500" {...props} />,
                                                            tr: ({ node, ...props }) => <tr className="hover:bg-gray-300 dark:hover:bg-gray-600" {...props} />,
                                                            th: ({ node, ...props }) => <th className="px-4 py-2 text-left font-medium" {...props} />,
                                                            td: ({ node, ...props }) => <td className="px-4 py-2" {...props} />,
                                                        }}
                                                    >
                                                        {streamingMessage.content}
                                                    </ReactMarkdown>
                                                )}

                                                {streamingMessage.isStreaming && (
                                                    <div className="typing-animation mt-2 inline-flex items-center text-gray-400">
                                                        <span></span>
                                                        <span></span>
                                                        <span></span>
                                                    </div>
                                                )}
                                            </div>
                                            <div className="text-xs mt-2 text-right text-gray-400/80">
                                                {new Date(streamingMessage.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    !isInitialLoading && loading.message && (
                                        <div className="flex justify-start items-end space-x-2">
                                            <div className="w-full max-w-3xl rounded-2xl px-4 py-2 assistant-message text-black dark:text-white rounded-bl-none">
                                                <div className="typing-animation inline-flex items-center text-gray-400">
                                                    <span></span>
                                                    <span></span>
                                                    <span></span>
                                                </div>
                                            </div>
                                        </div>
                                    )
                                )}
                                <div ref={messagesEndRef} />
                            </>
                        )}
                    </div>
                </div>

                <div className="flex-shrink-0 w-[95%] max-w-3xl mx-auto">
                    {isUploading && (
                        <div className="mb-2 px-2">
                            <div className="flex items-center p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-800/30">
                                <div className="flex-shrink-0 mr-3">
                                    <div className="w-8 h-8 flex items-center justify-center">
                                        <svg className="animate-spin w-5 h-5 text-blue-500 dark:text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                    </div>
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="text-sm font-medium text-blue-700 dark:text-blue-300">
                                        {uploadedFiles.length === 1
                                            ? `Uploading ${uploadedFiles[0]?.name}`
                                            : `Uploading ${uploadedFiles.length} files`}
                                    </div>
                                    <div className="mt-1 relative h-1.5 w-full bg-blue-100 dark:bg-blue-800/40 rounded-full overflow-hidden">
                                        <div
                                            className="absolute left-0 top-0 h-full bg-blue-500 dark:bg-blue-400 transition-all duration-300"
                                            style={{ width: `${uploadProgress}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {uploadedFiles.length > 0 && !isUploading && (
                        <div className="mb-2 flex flex-wrap gap-2">
                            {uploadedFiles.map((file, index) => (
                                <div
                                    key={`${file.name}-${index}`}
                                    className="flex items-center py-1 px-2 bg-gray-50 dark:bg-gray-800/50 rounded-md border border-gray-200 dark:border-gray-700/50 max-w-fit"
                                >
                                    <div className="mr-1.5 text-gray-500 dark:text-gray-400">
                                        {getFileIcon(file.name)}
                                    </div>
                                    <span className="text-xs font-medium text-gray-700 dark:text-gray-300 truncate max-w-[140px]">
                                        {file.name}
                                    </span>
                                    <button
                                        onClick={() => handleRemoveUploadedFile(index)}
                                        className="ml-1.5 text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 p-0.5 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700/50 transition-colors"
                                        aria-label="Remove file"
                                    >
                                        <IoClose size={14} />
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    <AdminMessageInput
                        onSubmit={handleChatSubmit}
                        onFileUpload={handleFileUpload}
                        isLoading={loading.message}
                        currentGptName={gptData?.name}
                        webSearchEnabled={webSearchEnabled}
                        setWebSearchEnabled={setWebSearchEnabled}
                        showWebSearchIcon={showWebSearchToggle}
                    />
                </div>

                {isProfileOpen && (
                    <div
                        className="fixed inset-0 z-20"
                        onClick={() => setIsProfileOpen(false)}
                    />
                )}
            </div>
        </>
    );
};

export default AdminChat;
