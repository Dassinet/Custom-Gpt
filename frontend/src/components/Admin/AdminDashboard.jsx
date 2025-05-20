import React, { useState, useRef, useEffect, useMemo, lazy, Suspense, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import AdminSidebar from './AdminSidebar';
const CreateCustomGpt = lazy(() => import('./CreateCustomGpt'));
const MoveToFolderModal = lazy(() => import('./MoveToFolderModal'));
import { FiSearch, FiChevronDown, FiChevronUp, FiMenu, FiPlus, FiGlobe, FiUsers, FiMessageSquare, FiGrid, FiList, FiEdit, FiTrash2, FiFolderPlus } from 'react-icons/fi';
import { SiOpenai, SiGooglegemini } from 'react-icons/si';
import { FaRobot } from 'react-icons/fa6';
import { BiLogoMeta } from 'react-icons/bi';
import { RiOpenaiFill, RiMoonFill, RiSunFill } from 'react-icons/ri';
import { TbRouter } from 'react-icons/tb';
import { axiosInstance } from '../../api/axiosInstance';
import { useTheme } from '../../context/ThemeContext';
import { toast } from 'react-toastify';

const defaultAgentImage = '/img.png';

// Model icons mapping
const modelIcons = {
    'router-engine': <TbRouter className="text-yellow-500" size={18} />,
    'gpt-4': <RiOpenaiFill className="text-green-500" size={18} />,
    'gpt-3.5': <SiOpenai className="text-green-400" size={16} />,
    'claude': <FaRobot className="text-purple-400" size={16} />,
    'gemini': <SiGooglegemini className="text-blue-400" size={16} />,
    'llama': <BiLogoMeta className="text-blue-500" size={18} />
};

// Add this function after the modelIcons declaration (around line 25)
const getDisplayModelName = (modelType) => {
    if (modelType === 'openrouter/auto') return 'router-engine';
    return modelType;
};

// Enhanced Agent Card component
const EnhancedAgentCard = ({ agent, onClick, onEdit, onDelete, onMoveToFolder }) => {
    const { isDarkMode } = useTheme();

    return (
        <div
            className="bg-white dark:bg-gray-800 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 hover:border-blue-400/50 dark:hover:border-gray-600 transition-all shadow-md hover:shadow-lg flex flex-col cursor-pointer group"
            onClick={onClick}
        >
            <div className="h-32 sm:h-36 bg-gradient-to-br from-gray-100 to-gray-300 dark:from-gray-700 dark:to-gray-900 relative flex-shrink-0 overflow-hidden">
                {agent.image ? (
                    <img
                        src={agent.image}
                        alt={agent.name}
                        className="w-full h-full object-cover object-center opacity-90 dark:opacity-80 group-hover:scale-105 transition-transform duration-300"
                        loading="lazy"
                        onError={(e) => {
                            e.target.onerror = null;
                            e.target.src = defaultAgentImage;
                        }}
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-blue-50/50 to-purple-100/50 dark:from-blue-900/30 dark:to-purple-900/30">
                        <span className={`text-3xl sm:text-4xl ${isDarkMode ? 'text-white/30' : 'text-gray-500/40'}`}>{agent.name.charAt(0)}</span>
                    </div>
                )}

                {/* Action Buttons - Added for edit, delete and move to folder */}
                <div className="absolute top-2 right-2 flex gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                    <button
                        onClick={(e) => { e.stopPropagation(); onMoveToFolder(agent); }}
                        className="p-1.5 sm:p-2 bg-white/80 dark:bg-gray-900/70 text-gray-700 dark:text-gray-200 rounded-full hover:bg-green-500 hover:text-white dark:hover:bg-green-700/80 transition-colors shadow"
                        title="Move to Folder"
                    >
                        <FiFolderPlus size={14} />
                    </button>
                    <button
                        onClick={(e) => { e.stopPropagation(); onEdit(agent.id); }}
                        className="p-1.5 sm:p-2 bg-white/80 dark:bg-gray-900/70 text-gray-700 dark:text-gray-200 rounded-full hover:bg-blue-500 hover:text-white dark:hover:bg-blue-700/80 transition-colors shadow"
                        title="Edit GPT"
                    >
                        <FiEdit size={14} />
                    </button>
                    <button
                        onClick={(e) => { e.stopPropagation(); onDelete(agent.id); }}
                        className="p-1.5 sm:p-2 bg-white/80 dark:bg-gray-900/70 text-gray-700 dark:text-gray-200 rounded-full hover:bg-red-500 hover:text-white dark:hover:bg-red-700/80 transition-colors shadow"
                        title="Delete GPT"
                    >
                        <FiTrash2 size={14} />
                    </button>
                </div>
            </div>

            <div className="p-3 sm:p-4 flex flex-col flex-grow">
                <div className="flex items-start justify-between mb-1.5 sm:mb-2">
                    <h3 className="font-semibold text-base sm:text-lg line-clamp-1 text-gray-900 dark:text-white">{agent.name}</h3>
                    <div className="flex items-center flex-shrink-0 gap-1 bg-gray-100 dark:bg-gray-700 px-1.5 sm:px-2 py-0.5 rounded text-[10px] sm:text-xs text-gray-600 dark:text-gray-300">
                        {React.cloneElement(modelIcons[agent.modelType === 'openrouter/auto' ? 'router-engine' : agent.modelType] || <FaRobot className="text-gray-500" />, { size: 12 })}
                        <span className="hidden sm:inline">{agent.modelType === 'openrouter/auto' ? 'router-engine' : agent.modelType}</span>
                    </div>
                </div>

                {agent.hasWebSearch && (
                    <div className="flex items-center gap-1 text-xs text-blue-500 dark:text-blue-400 mb-1">
                        <FiGlobe size={12} />
                        <span>Web search</span>
                    </div>
                )}


            </div>
        </div>
    );
};

const AdminDashboard = ({ userName = "Admin User" }) => {
    const [showCreateGpt, setShowCreateGpt] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [isSortOpen, setIsSortOpen] = useState(false);
    const [sortOption, setSortOption] = useState('Default');
    const sortOptions = ['Default', 'Latest', 'Older'];
    const dropdownRef = useRef(null);
    const [showSidebar, setShowSidebar] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [agentsData, setAgentsData] = useState({
        featured: [],
        productivity: [],
        education: [],
        entertainment: []
    });
    const [gptCreated, setGptCreated] = useState(false);
    const { isDarkMode, toggleTheme } = useTheme();
    const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'list'
    const navigate = useNavigate();
    // New state for move to folder modal
    const [showMoveModal, setShowMoveModal] = useState(false);
    const [agentToMove, setAgentToMove] = useState(null);
    const [folders, setFolders] = useState(['Uncategorized']);

    // Handler for deleting a GPT
    const handleDeleteGpt = useCallback(async (id) => {
        if (window.confirm("Are you sure you want to delete this GPT?")) {
            setLoading(true);
            try {
                const response = await axiosInstance.delete(`/api/custom-gpts/${id}`, { withCredentials: true });
                if (response.data.success) {
                    toast.success(`GPT deleted successfully.`);

                    // Update the state to reflect the deletion
                    const updatedData = {};
                    Object.keys(agentsData).forEach(category => {
                        updatedData[category] = agentsData[category].filter(agent => agent.id !== id);
                    });
                    setAgentsData(updatedData);
                } else {
                    toast.error(response.data.message || "Failed to delete GPT");
                }
            } catch (err) {
                console.error("Error deleting custom GPT:", err);
                toast.error(err.response?.data?.message || "Error deleting GPT");
            } finally {
                setLoading(false);
            }
        }
    }, [agentsData]);

    // Handler for editing a GPT
    const handleEditGpt = useCallback((id) => {
        navigate(`/admin/edit-gpt/${id}`);
    }, [navigate]);

    // Handler for moving a GPT to folder
    const handleMoveToFolder = useCallback((agent) => {
        setAgentToMove(agent);
        setShowMoveModal(true);
    }, []);

    // Handler for when a GPT is successfully moved
    const handleGptMoved = useCallback((movedGpt, newFolderName) => {
        // Update the folder info in state
        const updatedData = {};
        Object.keys(agentsData).forEach(category => {
            updatedData[category] = agentsData[category].map(agent =>
                agent.id === movedGpt._id ? { ...agent, folder: newFolderName || null } : agent
            );
        });
        setAgentsData(updatedData);

        // Add new folder to folders list if it doesn't exist yet
        if (newFolderName && !folders.includes(newFolderName)) {
            setFolders(prev => [...prev, newFolderName]);
        }

        setShowMoveModal(false);
        setAgentToMove(null);
        toast.success(`GPT moved successfully.`);
    }, [agentsData, folders]);

    const applySorting = (data, sortOpt) => {
        if (sortOpt === 'Default') return data;
        const sortedData = { ...data };
        const sortFn = sortOpt === 'Latest'
            ? (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
            : (a, b) => new Date(a.createdAt) - new Date(b.createdAt);
        Object.keys(sortedData).forEach(category => {
            if (Array.isArray(sortedData[category])) {
                sortedData[category] = [...sortedData[category]].sort(sortFn);
            }
        });
        return sortedData;
    };

    useEffect(() => {
        const fetchAgents = async () => {
            try {
                setLoading(true);
                const response = await axiosInstance.get(`/api/custom-gpts`, {
                    withCredentials: true
                });
                if (response.data.success && response.data.customGpts) {
                    const sortedGpts = [...response.data.customGpts].sort((a, b) =>
                        new Date(b.createdAt) - new Date(a.createdAt)
                    );

                    // Extract folders for the move to folder functionality
                    const uniqueFolders = [...new Set(sortedGpts
                        .filter(gpt => gpt.folder)
                        .map(gpt => gpt.folder))];
                    setFolders(prev => [...new Set(['Uncategorized', ...uniqueFolders])]);

                    const categorizedData = {
                        featured: [],
                        productivity: [],
                        education: [],
                        entertainment: []
                    };
                    categorizedData.featured = sortedGpts.slice(0, 4).map(gpt => ({
                        id: gpt._id,
                        image: gpt.imageUrl || defaultAgentImage,
                        name: gpt.name,
                        status: gpt.status || 'unknown',
                        modelType: gpt.model,
                        hasWebSearch: gpt.capabilities?.webBrowsing,
                        createdAt: gpt.createdAt,
                        folder: gpt.folder // Add folder information
                    }));
                    sortedGpts.forEach(gpt => {
                        const text = (gpt.description + ' ' + gpt.name).toLowerCase();
                        const agent = {
                            id: gpt._id,
                            image: gpt.imageUrl || defaultAgentImage,
                            name: gpt.name,
                            status: gpt.status || 'unknown',
                            modelType: gpt.model,
                            hasWebSearch: gpt.capabilities?.webBrowsing,
                            createdAt: gpt.createdAt,
                            folder: gpt.folder // Add folder information
                        };
                        if (categorizedData.featured.some(a => a.name === gpt.name)) {
                            return;
                        }
                        if (text.includes('work') || text.includes('task') || text.includes('productivity')) {
                            categorizedData.productivity.push(agent);
                        } else if (text.includes('learn') || text.includes('study') || text.includes('education')) {
                            categorizedData.education.push(agent);
                        } else if (text.includes('game') || text.includes('movie') || text.includes('fun')) {
                            categorizedData.entertainment.push(agent);
                        } else {
                            const categories = ['productivity', 'education', 'entertainment'];
                            const randomCategory = categories[Math.floor(Math.random() * categories.length)];
                            categorizedData[randomCategory].push(agent);
                        }
                    });
                    setAgentsData(categorizedData);
                } else {
                    setError(response.data.message || "Failed to load agents data: Invalid response format");
                }
            } catch (err) {
                console.error("Error fetching agents:", err);
                setError(`Failed to load agents data. ${err.response?.data?.message || err.message || ''}`);
            } finally {
                setLoading(false);
            }
        };
        fetchAgents();
    }, [gptCreated]);

    useEffect(() => {
        const handleResize = () => {
            if (window.innerWidth >= 640) {
                setShowSidebar(false);
            }
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const filteredAgentsData = useMemo(() => {
        const searchTermLower = searchTerm.toLowerCase().trim();
        if (!searchTermLower) {
            return applySorting(agentsData, sortOption);
        }
        const filtered = {};
        Object.keys(agentsData).forEach(category => {
            filtered[category] = agentsData[category].filter(agent =>
                agent.name.toLowerCase().includes(searchTermLower) ||
                (agent.modelType && agent.modelType.toLowerCase().includes(searchTermLower))
            );
        });
        return applySorting(filtered, sortOption);
    }, [searchTerm, agentsData, sortOption]);

    useEffect(() => {
        function handleClickOutside(event) {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsSortOpen(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [dropdownRef]);

    const handleSortChange = (option) => {
        setSortOption(option);
        setIsSortOpen(false);
    };

    const handleNavigateToChat = (agentId) => {
        navigate(`/admin/chat/${agentId}`);
    };

    const hasSearchResults = Object.values(filteredAgentsData).some(
        category => category.length > 0
    );

    if (loading) {
        return (
            <div className="flex h-screen bg-white dark:bg-black text-black dark:text-white items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex h-screen bg-white dark:bg-black text-black dark:text-white items-center justify-center">
                <div className="text-center p-4">
                    <p className="text-red-400 mb-4">{error}</p>
                    <button
                        onClick={() => window.location.reload()}
                        className="bg-blue-500 text-white px-6 py-2 rounded-full font-medium hover:bg-blue-600 transition-all"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-screen bg-gray-50 dark:bg-gray-900 text-black dark:text-white font-sans">
            {/* Mobile Sidebar Overlay */}
            {showSidebar && (
                <div
                    className="fixed inset-0 bg-black/80 z-40 sm:hidden"
                    onClick={() => setShowSidebar(false)}
                />
            )}

            {/* Main Content */}
            <div className="flex-1 flex flex-col h-full overflow-hidden">
                {!showCreateGpt ? (
                    <>
                        {/* Header Section */}
                        <header className="bg-white dark:bg-black px-4 sm:px-8 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0 shadow-sm">
                            {/* Desktop Header */}
                            <div className="hidden sm:flex items-center justify-between">
                                <div className="flex items-center">
                                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Admin Dashboard</h1>
                                    <div className="flex items-center ml-4 gap-2">
                                        <button
                                            onClick={() => setViewMode('grid')}
                                            className={`p-2 rounded-md ${viewMode === 'grid' ? 'bg-gray-200 dark:bg-gray-700' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
                                            title="Grid View"
                                        >
                                            <FiGrid className="text-gray-700 dark:text-gray-300" />
                                        </button>
                                        <button
                                            onClick={() => setViewMode('list')}
                                            className={`p-2 rounded-md ${viewMode === 'list' ? 'bg-gray-200 dark:bg-gray-700' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
                                            title="List View"
                                        >
                                            <FiList className="text-gray-700 dark:text-gray-300" />
                                        </button>
                                    </div>
                                </div>
                                <div className="flex items-center gap-4">
                                    <div className="relative">
                                        <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                                        <input
                                            type="text"
                                            placeholder="Search GPTs..."
                                            value={searchTerm}
                                            onChange={(e) => setSearchTerm(e.target.value)}
                                            className="w-64 pl-10 pr-4 py-2 rounded-md bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                                        />
                                    </div>
                                    <button
                                        onClick={toggleTheme}
                                        className="p-2 rounded-full bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                                        title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
                                    >
                                        {isDarkMode ? <RiSunFill size={20} className="text-yellow-400" /> : <RiMoonFill size={20} className="text-gray-700" />}
                                    </button>
                                    <button
                                        onClick={() => setShowCreateGpt(true)}
                                        className="flex items-center gap-2 bg-black dark:bg-white text-white dark:text-black px-4 py-2 rounded-md font-medium transition-colors"
                                    >
                                        <FiPlus size={18} />
                                        Create GPT
                                    </button>
                                </div>
                            </div>
                            {/* Mobile Header */}
                            <div className="block sm:hidden">
                                <div className="flex items-center mb-4">
                                    <button
                                        onClick={() => setShowSidebar(!showSidebar)}
                                        className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
                                    >
                                        <FiMenu size={24} className="text-gray-700 dark:text-gray-300" />
                                    </button>
                                    <h1 className="flex-1 text-center text-xl font-bold text-gray-900 dark:text-white">Admin Dashboard</h1>
                                    <button
                                        onClick={toggleTheme}
                                        className="p-2 rounded-full bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600"
                                    >
                                        {isDarkMode ? <RiSunFill size={20} className="text-yellow-400" /> : <RiMoonFill size={20} className="text-gray-700" />}
                                    </button>
                                </div>
                                <div className="flex items-center gap-3">
                                    <div className="flex-1 relative">
                                        <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                                        <input
                                            type="text"
                                            placeholder="Search GPTs..."
                                            value={searchTerm}
                                            onChange={(e) => setSearchTerm(e.target.value)}
                                            className="w-full pl-10 pr-4 py-2 rounded-md bg-gray-100 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                                        />
                                    </div>
                                    <button
                                        onClick={() => setShowCreateGpt(true)}
                                        className="bg-black dark:bg-white text-white dark:text-black p-2 rounded-md"
                                    >
                                        <FiPlus size={24} />
                                    </button>
                                </div>
                                <div className="flex justify-center mt-3 gap-2">
                                    <button
                                        onClick={() => setViewMode('grid')}
                                        className={`p-2 rounded-md ${viewMode === 'grid' ? 'bg-gray-200 dark:bg-gray-700' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
                                    >
                                        <FiGrid className="text-gray-700 dark:text-gray-300" />
                                    </button>
                                    <button
                                        onClick={() => setViewMode('list')}
                                        className={`p-2 rounded-md ${viewMode === 'list' ? 'bg-gray-200 dark:bg-gray-700' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
                                    >
                                        <FiList className="text-gray-700 dark:text-gray-300" />
                                    </button>
                                </div>
                            </div>
                        </header>

                        {/* Main Content Area - With hidden scrollbar styling */}
                        <div className="flex-1 flex flex-col p-4 sm:p-6 overflow-y-auto bg-gray-50 dark:bg-black scrollbar-hide [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
                            {searchTerm && !hasSearchResults ? (
                                <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                                    No agents found for "{searchTerm}"
                                </div>
                            ) : (
                                <>
                                    {/* Featured Agents Section */}
                                    {filteredAgentsData.featured && filteredAgentsData.featured.length > 0 && (
                                        <div className="mb-8 flex-shrink-0">
                                            <div className="flex items-center justify-between mb-4">
                                                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Featured Agents</h2>
                                                <div className="relative" ref={dropdownRef}>
                                                    <button
                                                        onClick={() => setIsSortOpen(!isSortOpen)}
                                                        className="flex items-center text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white py-1.5 px-3 bg-white dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-700 shadow-sm"
                                                    >
                                                        Sort: {sortOption}
                                                        {isSortOpen ? <FiChevronUp className="ml-2" /> : <FiChevronDown className="ml-2" />}
                                                    </button>
                                                    {isSortOpen && (
                                                        <div className="absolute top-full right-0 mt-1 w-36 bg-white dark:bg-gray-800 rounded-md shadow-lg z-10 border border-gray-200 dark:border-gray-700">
                                                            <ul>
                                                                {sortOptions.map((option) => (
                                                                    <li key={option}>
                                                                        <button
                                                                            onClick={() => handleSortChange(option)}
                                                                            className={`block w-full text-left px-4 py-2 text-sm ${sortOption === option ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'} transition-all`}
                                                                        >
                                                                            {option}
                                                                        </button>
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>

                                            <div className={viewMode === 'grid' ?
                                                "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4" :
                                                "space-y-3"
                                            }>
                                                {filteredAgentsData.featured.map((agent) => (
                                                    viewMode === 'grid' ? (
                                                        <EnhancedAgentCard
                                                            key={agent.id || agent.name}
                                                            agent={agent}
                                                            onClick={() => handleNavigateToChat(agent.id)}
                                                            onEdit={handleEditGpt}
                                                            onDelete={handleDeleteGpt}
                                                            onMoveToFolder={handleMoveToFolder}
                                                        />
                                                    ) : (
                                                        <div
                                                            key={agent.id || agent.name}
                                                            className="flex items-center bg-white dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-400/50 dark:hover:border-gray-600 shadow-sm cursor-pointer group"
                                                            onClick={() => handleNavigateToChat(agent.id)}
                                                        >
                                                            <div className="h-14 w-14 bg-gradient-to-br from-gray-100 to-gray-300 dark:from-gray-700 dark:to-gray-900 rounded-md overflow-hidden mr-4 flex-shrink-0">
                                                                {agent.image ? (
                                                                    <img src={agent.image} alt={agent.name} className="w-full h-full object-cover" />
                                                                ) : (
                                                                    <div className="w-full h-full flex items-center justify-center">
                                                                        <span className="text-xl text-gray-500/40 dark:text-white/30">{agent.name.charAt(0)}</span>
                                                                    </div>
                                                                )}
                                                            </div>
                                                            <div className="flex-grow min-w-0">
                                                                <div className="flex items-center justify-between">
                                                                    <h3 className="font-semibold text-sm text-gray-900 dark:text-white truncate">{agent.name}</h3>
                                                                    <div className="flex items-center ml-2 bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded text-[10px] text-gray-600 dark:text-gray-300">
                                                                        {React.cloneElement(modelIcons[agent.modelType === 'openrouter/auto' ? 'router-engine' : agent.modelType] || <FaRobot className="text-gray-500" />, { size: 12 })}
                                                                        <span className="ml-1">{agent.modelType === 'openrouter/auto' ? 'router-engine' : agent.modelType}</span>
                                                                    </div>
                                                                </div>

                                                            </div>

                                                            {/* Add action buttons for list view */}
                                                            <div className="flex gap-1 ml-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                                <button
                                                                    onClick={(e) => { e.stopPropagation(); handleMoveToFolder(agent); }}
                                                                    className="p-1.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-green-500 hover:text-white transition-colors"
                                                                    title="Move to Folder"
                                                                >
                                                                    <FiFolderPlus size={14} />
                                                                </button>
                                                                <button
                                                                    onClick={(e) => { e.stopPropagation(); handleEditGpt(agent.id); }}
                                                                    className="p-1.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-blue-500 hover:text-white transition-colors"
                                                                    title="Edit GPT"
                                                                >
                                                                    <FiEdit size={14} />
                                                                </button>
                                                                <button
                                                                    onClick={(e) => { e.stopPropagation(); handleDeleteGpt(agent.id); }}
                                                                    className="p-1.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-red-500 hover:text-white transition-colors"
                                                                    title="Delete GPT"
                                                                >
                                                                    <FiTrash2 size={14} />
                                                                </button>
                                                            </div>
                                                        </div>
                                                    )
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Categories Header */}
                                    <h2 className="text-xl font-semibold mb-6 flex-shrink-0 text-gray-900 dark:text-white">Categories</h2>

                                    {/* Scrollable Categories */}
                                    <div className="space-y-8">
                                        {Object.entries(filteredAgentsData).map(([category, agents]) => {
                                            if (category === 'featured' || agents.length === 0) return null;
                                            const categoryTitle = category
                                                .replace(/([A-Z])/g, ' $1')
                                                .replace(/^./, (str) => str.toUpperCase());

                                            return (
                                                <div key={category} className="mb-8">
                                                    <div className="flex items-center justify-between mb-3">
                                                        <h3 className="text-lg font-medium text-gray-800 dark:text-gray-200">{categoryTitle}</h3>
                                                        <span className="text-sm text-gray-500 dark:text-gray-400">{agents.length} {agents.length === 1 ? 'agent' : 'agents'}</span>
                                                    </div>

                                                    <div className={viewMode === 'grid' ?
                                                        "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4" :
                                                        "space-y-3"
                                                    }>
                                                        {agents.map((agent) => (
                                                            viewMode === 'grid' ? (
                                                                <EnhancedAgentCard
                                                                    key={agent.id || agent.name}
                                                                    agent={agent}
                                                                    onClick={() => handleNavigateToChat(agent.id)}
                                                                    onEdit={handleEditGpt}
                                                                    onDelete={handleDeleteGpt}
                                                                    onMoveToFolder={handleMoveToFolder}
                                                                />
                                                            ) : (
                                                                <div
                                                                    key={agent.id || agent.name}
                                                                    className="flex items-center bg-white dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-400/50 dark:hover:border-gray-600 shadow-sm cursor-pointer group"
                                                                    onClick={() => handleNavigateToChat(agent.id)}
                                                                >
                                                                    <div className="h-14 w-14 bg-gradient-to-br from-gray-100 to-gray-300 dark:from-gray-700 dark:to-gray-900 rounded-md overflow-hidden mr-4 flex-shrink-0">
                                                                        {agent.image ? (
                                                                            <img src={agent.image} alt={agent.name} className="w-full h-full object-cover" />
                                                                        ) : (
                                                                            <div className="w-full h-full flex items-center justify-center">
                                                                                <span className="text-xl text-gray-500/40 dark:text-white/30">{agent.name.charAt(0)}</span>
                                                                            </div>
                                                                        )}
                                                                    </div>
                                                                    <div className="flex-grow min-w-0">
                                                                        <div className="flex items-center justify-between">
                                                                            <h3 className="font-semibold text-sm text-gray-900 dark:text-white truncate">{agent.name}</h3>
                                                                            <div className="flex items-center ml-2 bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded text-[10px] text-gray-600 dark:text-gray-300">
                                                                                {React.cloneElement(modelIcons[agent.modelType === 'openrouter/auto' ? 'router-engine' : agent.modelType] || <FaRobot className="text-gray-500" />, { size: 12 })}
                                                                                <span className="ml-1">{agent.modelType === 'openrouter/auto' ? 'router-engine' : agent.modelType}</span>
                                                                            </div>
                                                                        </div>
                                                                    </div>

                                                                    {/* Action buttons for list view */}
                                                                    <div className="flex gap-1 ml-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                                        <button
                                                                            onClick={(e) => { e.stopPropagation(); handleMoveToFolder(agent); }}
                                                                            className="p-1.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-green-500 hover:text-white transition-colors"
                                                                            title="Move to Folder"
                                                                        >
                                                                            <FiFolderPlus size={14} />
                                                                        </button>
                                                                        <button
                                                                            onClick={(e) => { e.stopPropagation(); handleEditGpt(agent.id); }}
                                                                            className="p-1.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-blue-500 hover:text-white transition-colors"
                                                                            title="Edit GPT"
                                                                        >
                                                                            <FiEdit size={14} />
                                                                        </button>
                                                                        <button
                                                                            onClick={(e) => { e.stopPropagation(); handleDeleteGpt(agent.id); }}
                                                                            className="p-1.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-red-500 hover:text-white transition-colors"
                                                                            title="Delete GPT"
                                                                        >
                                                                            <FiTrash2 size={14} />
                                                                        </button>
                                                                    </div>
                                                                </div>
                                                            )
                                                        ))}
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </>
                            )}
                        </div>
                    </>
                ) : (
                    <div className="h-full">
                        <Suspense fallback={<div className="flex h-full items-center justify-center text-gray-500 dark:text-gray-400">Loading Editor...</div>}>
                            <CreateCustomGpt
                                onGoBack={() => setShowCreateGpt(false)}
                                onGptCreated={() => {
                                    setGptCreated(prev => !prev);
                                    setShowCreateGpt(false);
                                }}
                            />
                        </Suspense>
                    </div>
                )}
            </div>

            {/* Move to Folder Modal */}
            {showMoveModal && agentToMove && (
                <Suspense fallback={<div className="fixed inset-0 bg-black/50 flex items-center justify-center text-white">Loading...</div>}>
                    <MoveToFolderModal
                        isOpen={showMoveModal}
                        onClose={() => { setShowMoveModal(false); setAgentToMove(null); }}
                        gpt={{
                            _id: agentToMove.id,
                            name: agentToMove.name,
                            folder: agentToMove.folder
                        }}
                        existingFolders={folders.filter(f => f !== 'Uncategorized')}
                        onSuccess={handleGptMoved}
                    />
                </Suspense>
            )}
        </div>
    );
};

export default AdminDashboard;