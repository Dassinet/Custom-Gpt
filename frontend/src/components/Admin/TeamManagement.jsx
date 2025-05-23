import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
    FiSearch,
    FiFilter,
    FiMoreVertical,
    FiUser,
    FiUsers,
    FiBell,
    FiBox,
    FiCalendar,
    FiMail,
    FiEdit,
    FiTrash2,
    FiChevronRight,
    FiChevronDown,
    FiCheck,
    FiSun,
    FiMoon
} from 'react-icons/fi';
import AssignGptsModal from './AssignGptsModal';
import TeamMemberDetailsModal from './TeamMemberDetailsModal';
import InviteTeamMemberModal from './InviteTeamMemberModal';
import EditPermissionsModal from './EditPermissionsModal';
import { axiosInstance } from '../../api/axiosInstance';
import { toast } from 'react-toastify';
import { useTheme } from '../../context/ThemeContext';
import { useAuth } from '../../context/AuthContext';
import { AutoSizer, List } from 'react-virtualized';
import debounce from 'lodash/debounce';

// List of departments for filter dropdown
const departments = [
    'All Departments',
    'Product',
    'Engineering',
    'Design',
    'Marketing',
    'Sales',
    'Customer Support'
];

// Memoized MobileTeamMemberCard
const MobileTeamMemberCard = React.memo(({ member, onViewDetails, isCurrentUser }) => (
    <div
        className={`bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700 mb-3 ${isCurrentUser ? 'opacity-80' : 'cursor-pointer'}`}
        onClick={() => !isCurrentUser && onViewDetails(member)}
    >
        <div className="flex items-center justify-between mb-3">
            <div className="flex items-center">
                <div className="w-10 h-10 rounded-full bg-gray-200 dark:bg-gray-600 flex items-center justify-center mr-3">
                    <FiUser className="text-gray-600 dark:text-gray-300" />
                </div>
                <div>
                    <p className="font-semibold text-gray-900 dark:text-white">{member.name}</p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">{member.email}</p>
                    {isCurrentUser && (
                        <span className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 px-2 py-0.5 rounded-full mt-1 inline-block">
                            You
                        </span>
                    )}
                </div>
            </div>
            {isCurrentUser ? (
                <span className="text-xs text-gray-400 italic">Current user</span>
            ) : (
                <FiChevronRight className="text-gray-400 dark:text-gray-500" />
            )}
        </div>
        <div className="text-sm space-y-1">
            <p><strong className="text-gray-600 dark:text-gray-300">Role:</strong> {member.role}</p>
            <p><strong className="text-gray-600 dark:text-gray-300">Department:</strong> {member.department}</p>
            <p><strong className="text-gray-600 dark:text-gray-300">Status:</strong>
                <span className={`ml-1 px-2 py-0.5 rounded-full text-xs font-medium ${member.status === 'Active'
                    ? 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300'
                    : 'bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300'}`}>
                    {member.status}
                </span>
            </p>
            <p><strong className="text-gray-600 dark:text-gray-300">GPTs:</strong> {member.assignedGPTs}</p>
        </div>
    </div>
));

const TeamManagement = () => {
    const [teamMembers, setTeamMembers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedMember, setSelectedMember] = useState(null);
    const [showActionsMenu, setShowActionsMenu] = useState(null);
    const [showDepartmentsDropdown, setShowDepartmentsDropdown] = useState(false);
    const [showStatusDropdown, setShowStatusDropdown] = useState(false);
    const [selectedDepartment, setSelectedDepartment] = useState('All Departments');
    const [selectedStatus, setSelectedStatus] = useState('All Status');
    const [isMobileView, setIsMobileView] = useState(window.innerWidth < 768);
    const [showAssignGptsModal, setShowAssignGptsModal] = useState(false);
    const [selectedMemberForGpts, setSelectedMemberForGpts] = useState(null);
    const [showDetailsModal, setShowDetailsModal] = useState(false);
    const [selectedMemberForDetails, setSelectedMemberForDetails] = useState(null);
    const [showInviteModal, setShowInviteModal] = useState(false);
    const [pendingInvitesCount, setPendingInvitesCount] = useState(0);
    const [assignmentChanged, setAssignmentChanged] = useState(false);
    const [refreshInterval, setRefreshInterval] = useState(null);
    const [showEditPermissionsModal, setShowEditPermissionsModal] = useState(false);
    const [selectedMemberForPermissions, setSelectedMemberForPermissions] = useState(null);
    const { isDarkMode, toggleTheme } = useTheme();
    const { user } = useAuth();
    const actionsMenuRef = useRef(null);
    const departmentFilterRef = useRef(null);
    const statusFilterRef = useRef(null);
    const [page, setPage] = useState(1);
    const [pageSize, setPageSize] = useState(10);
    const [cachedMembers, setCachedMembers] = useState({});
    const [totalMembers, setTotalMembers] = useState(0);
    const abortControllerRef = useRef(null);

    // Responsive detection
    useEffect(() => {
        const handleResize = () => {
            setIsMobileView(window.innerWidth < 768);
        };

        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // Error handling
    const handleApiError = (error, defaultMessage) => {
        console.error(defaultMessage, error);
        const errorMessage = error.response?.data?.message || defaultMessage;
        toast?.error(errorMessage);
        return errorMessage;
    };

    // Format date
    const formatDate = (dateString) => {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
        });
    };

    // Debounced fetchTeamData
    const debouncedFetchTeamData = useMemo(
        () =>
            debounce(async (pageNum, refresh) => {
                if (abortControllerRef.current) {
                    abortControllerRef.current.abort();
                }
                abortControllerRef.current = new AbortController();

                if (!refresh && cachedMembers[pageNum]) {
                    setTeamMembers(cachedMembers[pageNum]);
                    setLoading(false);
                    return;
                }

                try {
                    setLoading(true);
                    const response = await axiosInstance.get(`/api/auth/users/with-gpt-counts?page=${pageNum}&limit=${pageSize}`, {
                        withCredentials: true,
                        signal: abortControllerRef.current.signal,
                    });

                    if (response.data && response.data.users) {
                        const formattedUsers = response.data.users.map(user => {
                            const isActive = user.lastActive
                                ? (new Date() - new Date(user.lastActive)) < 24 * 60 * 60 * 1000
                                : false;

                            return {
                                id: user._id,
                                name: user.name,
                                email: user.email,
                                role: user.role || 'Employee',
                                department: user.department || 'Not Assigned',
                                joined: formatDate(user.createdAt),
                                lastActive: user.lastActive ? formatDate(user.lastActive) : 'Never',
                                status: isActive ? 'Active' : 'Inactive',
                                assignedGPTs: user.gptCount || 0,
                            };
                        });

                        setTeamMembers(formattedUsers);
                        setTotalMembers(response.data.total || formattedUsers.length);
                        setError(null);

                        setCachedMembers(prev => ({
                            ...prev,
                            [pageNum]: formattedUsers,
                        }));
                    }
                } catch (err) {
                    if (err.name === 'AbortError') return;
                    console.error("Error fetching team members:", err);
                    setError("Failed to load team data. Please check your connection.");
                } finally {
                    setLoading(false);
                }
            }, 300),
        [pageSize, cachedMembers]
    );

    // Fetch team data
    const fetchTeamData = useCallback(
        async (refresh = false) => {
            debouncedFetchTeamData(page, refresh);
        },
        [page, debouncedFetchTeamData]
    );

    // Initial mount effect
    useEffect(() => {
        fetchTeamData();
        return () => {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, [fetchTeamData]);

    // Dynamic polling
    useEffect(() => {
        const interval = setInterval(() => {
            if (document.visibilityState === 'visible' && !loading) {
                fetchTeamData(true);
            }
        }, 60000);

        return () => clearInterval(interval);
    }, [fetchTeamData, loading]);

    // Handle GPT assignment changes
    const handleGptAssignmentChange = useCallback((memberId) => {
        if (typeof memberId !== 'string' || !/^[0-9a-fA-F]{24}$/.test(memberId)) {
            console.error("Invalid member ID:", memberId);
            return;
        }

        const fetchUpdatedCount = async () => {
            try {
                const response = await axiosInstance.get(`/api/auth/users/${memberId}/gpt-count`, {
                    withCredentials: true,
                });

                if (response.data && typeof response.data.count !== 'undefined') {
                    setTeamMembers(prev =>
                        prev.map(member =>
                            member.id === memberId
                                ? { ...member, assignedGPTs: response.data.count }
                                : member
                        )
                    );

                    setCachedMembers(prev => {
                        const newCache = { ...prev };
                        Object.keys(newCache).forEach(pageKey => {
                            if (Array.isArray(newCache[pageKey])) {
                                newCache[pageKey] = newCache[pageKey].map(member =>
                                    member.id === memberId
                                        ? { ...member, assignedGPTs: response.data.count }
                                        : member
                                );
                            }
                        });
                        return newCache;
                    });
                }
            } catch (err) {
                console.error("Error updating GPT count:", err);
            }
        };

        fetchUpdatedCount();
    }, []);

    // Fetch pending invites count
    useEffect(() => {
        const fetchPendingInvites = async () => {
            try {
                const response = await axiosInstance.get(`/api/auth/pending-invites/count`, {
                    withCredentials: true,
                });

                if (response.data && response.data.count !== undefined) {
                    setPendingInvitesCount(response.data.count);
                }
            } catch (err) {
                console.error("Error fetching pending invites count:", err);
            }
        };

        fetchPendingInvites();
    }, []);

    // Memoized filtered members
    const filteredMembers = useMemo(() => {
        return teamMembers.filter(member => {
            const matchesSearch =
                member.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                member.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
                (member.department && member.department.toLowerCase().includes(searchTerm.toLowerCase())) ||
                (member.position && member.position.toLowerCase().includes(searchTerm.toLowerCase()));

            const matchesDepartment =
                selectedDepartment === 'All Departments' ||
                member.department === selectedDepartment;

            const matchesStatus =
                selectedStatus === 'All Status' ||
                member.status === selectedStatus;

            return matchesSearch && matchesDepartment && matchesStatus;
        });
    }, [teamMembers, searchTerm, selectedDepartment, selectedStatus]);

    const toggleActionsMenu = (memberId) => {
        setShowActionsMenu(showActionsMenu === memberId ? null : memberId);
    };

    const handleInviteMember = () => {
        setShowInviteModal(true);
    };

    const handleAssignGpts = (member) => {
        setSelectedMemberForGpts(member);
        setShowAssignGptsModal(true);
        setShowActionsMenu(null);
    };

    const handleViewMemberDetails = (member) => {
        if (user?._id === member.id) return;
        setSelectedMemberForDetails(member);
        setShowDetailsModal(true);
    };

    // CSS for hiding scrollbars
    const scrollbarHideStyles = `
        .hide-scrollbar::-webkit-scrollbar {
            display: none;
        }
        .hide-scrollbar {
            -ms-overflow-style: none;
            scrollbar-width: none;
        }
        /* NEW: Apply hide-scrollbar to react-virtualized List */
        .ReactVirtualized__List::-webkit-scrollbar {
            display: none;
        }
        .ReactVirtualized__List {
            -ms-overflow-style: none;
            scrollbar-width: none;
        }
    `;

    const handleEmailTeamMember = async (email) => {
        window.location.href = `mailto:${email}`;
        setShowActionsMenu(null);
    };

    const handleEditPermissions = (member) => {
        setSelectedMemberForPermissions(member);
        setShowEditPermissionsModal(true);
        setShowActionsMenu(null);
    };

    const handleRemoveTeamMember = async (memberId) => {
        if (window.confirm("Are you sure you want to remove this team member? All their data including chat histories and assignments will be permanently deleted.")) {
            try {
                setLoading(true);
                const response = await axiosInstance.delete(`/api/auth/users/${memberId}`, {
                    withCredentials: true,
                });

                if (response.data.success) {
                    setTeamMembers(prev => prev.filter(member => member.id !== memberId));
                    setCachedMembers(prev => {
                        const newCache = { ...prev };
                        for (const pageKey in newCache) {
                            if (newCache[pageKey]) {
                                newCache[pageKey] = newCache[pageKey].filter(member => member.id !== memberId);
                            }
                        }
                        return newCache;
                    });
                    setTotalMembers(prev => Math.max(0, prev - 1));
                    toast.success("Team member and all associated data removed successfully");
                }
            } catch (err) {
                handleApiError(err, "Failed to remove team member");
            } finally {
                setLoading(false);
                setShowActionsMenu(null);
            }
        } else {
            setShowActionsMenu(null);
        }
    };

    const handlePermissionsUpdated = (updatedMember) => {
        setTeamMembers(prev =>
            prev.map(member =>
                member.id === updatedMember.id ? updatedMember : member
            )
        );
    };

    // Virtualized row renderer for mobile view
    const renderMobileRow = ({ index, key, style }) => {
        const member = filteredMembers[index];
        return (
            <div key={key} style={style}>
                <MobileTeamMemberCard
                    member={member}
                    onViewDetails={handleViewMemberDetails}
                    isCurrentUser={user?._id === member.id}
                />
            </div>
        );
    };

    // Pagination controls
    const renderPagination = () => {
        const totalPages = Math.ceil(totalMembers / pageSize);

        return (
            <div className="flex items-center justify-between border-t border-gray-200 dark:border-gray-700 px-4 py-3 sm:px-6 mt-4">
                <div className="flex-1 flex justify-between sm:hidden">
                    <button
                        onClick={() => setPage(Math.max(1, page - 1))}
                        disabled={page === 1}
                        className="relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50"
                    >
                        Previous
                    </button>
                    <button
                        onClick={() => setPage(Math.min(totalPages, page + 1))}
                        disabled={page === totalPages}
                        className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 text-sm font-medium rounded-md text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50"
                    >
                        Next
                    </button>
                </div>
            </div>
        );
    };

    // Skeleton loading
    const renderSkeleton = () => (
        <div className="space-y-3">
            {Array.from({ length: pageSize }).map((_, index) => (
                <div key={index} className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow border border-gray-200 dark:border-gray-700 animate-pulse">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center">
                            <div className="w-10 h-10 rounded-full bg-gray-200 dark:bg-gray-600 mr-3"></div>
                            <div>
                                <div className="h-4 bg-gray-200 dark:bg-gray-600 rounded w-24 mb-2"></div>
                                <div className="h-3 bg-gray-200 dark:bg-gray-600 rounded w-32"></div>
                            </div>
                        </div>
                        <div className="h-4 bg-gray-200 dark:bg-gray-600 rounded w-12"></div>
                    </div>
                    <div className="text-sm space-y-2">
                        <div className="h-3 bg-gray-200 dark:bg-gray-600 rounded w-40"></div>
                        <div className="h-3 bg-gray-200 dark:bg-gray-600 rounded w-48"></div>
                        <div className="h-3 bg-gray-200 dark:bg-gray-600 rounded w-32"></div>
                    </div>
                </div>
            ))}
        </div>
    );

    if (error) {
        return <div className="flex justify-center items-center h-screen bg-white dark:bg-black text-red-500 p-4">{error}</div>;
    }

    return (
        <div className="flex flex-col h-full bg-gray-50 dark:bg-black text-black dark:text-white p-4 sm:p-6 overflow-hidden">
            <style>{scrollbarHideStyles}</style>
            <div className="mb-6 flex-shrink-0 flex flex-col sm:flex-row sm:items-center sm:justify-between">
                <div className="text-center sm:text-left">
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">Team Management</h1>
                    <p className="text-gray-600 dark:text-gray-400">Manage your team members, permissions, and GPT assignments.</p>
                </div>
                <button
                    onClick={toggleTheme}
                    className={`p-2 rounded-full transition-colors self-center sm:self-auto mt-3 sm:mt-0 ${
                        isDarkMode 
                            ? 'bg-gray-800 text-yellow-400 hover:bg-gray-700' 
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                    aria-label={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
                    title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
                >
                    {isDarkMode ? <FiSun size={20} /> : <FiMoon size={20} />}
                </button>
            </div>

            <div className="mb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 flex-shrink-0">
                <div className="relative flex-grow sm:flex-grow-0 sm:w-64 md:w-72">
                    <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500" />
                    <input
                        type="text"
                        placeholder="Search members..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none text-black dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                    />
                </div>

                <div className="flex items-center gap-3">
                    <div className="relative" ref={departmentFilterRef}>
                        <button
                            onClick={() => setShowDepartmentsDropdown(!showDepartmentsDropdown)}
                            className="flex items-center gap-1 px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                        >
                            <FiFilter size={14} />
                            <span>{selectedDepartment === 'All Departments' ? 'Department' : selectedDepartment}</span>
                            <FiChevronDown size={16} className={`transition-transform ${showDepartmentsDropdown ? 'rotate-180' : ''}`} />
                        </button>
                        {showDepartmentsDropdown && (
                            <div className="absolute left-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 dark:ring-gray-700 z-10 overflow-hidden">
                                {departments.map(dept => (
                                    <button
                                        key={dept}
                                        onClick={() => { setSelectedDepartment(dept); setShowDepartmentsDropdown(false); }}
                                        className={`w-full text-left px-4 py-2 text-sm flex items-center justify-between ${selectedDepartment === dept ? 'font-semibold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30' : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'}`}
                                    >
                                        {dept}
                                        {selectedDepartment === dept && <FiCheck size={14} />}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                    <div className="relative" ref={statusFilterRef}>
                        <button
                            onClick={() => setShowStatusDropdown(!showStatusDropdown)}
                            className="flex items-center gap-1 px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
                        >
                            <FiUsers size={14} />
                            <span>{selectedStatus}</span>
                            <FiChevronDown size={16} className={`transition-transform ${showStatusDropdown ? 'rotate-180' : ''}`} />
                        </button>
                        {showStatusDropdown && (
                            <div className="absolute left-0 mt-2 w-40 bg-white dark:bg-gray-800 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 dark:ring-gray-700 z-10 overflow-hidden">
                                {['All Status', 'Active', 'Inactive'].map(status => (
                                    <button
                                        key={status}
                                        onClick={() => { setSelectedStatus(status); setShowStatusDropdown(false); }}
                                        className={`w-full text-left px-4 py-2 text-sm flex items-center justify-between ${selectedStatus === status ? 'font-semibold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30' : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'}`}
                                    >
                                        {status}
                                        {selectedStatus === status && <FiCheck size={14} />}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    <button
                        onClick={handleInviteMember}
                        className="flex items-center gap-2 px-4 py-2 bg-black/80 dark:bg-white hover:bg-black/80 dark:hover:bg-white/70 text-white dark:text-black rounded-lg font-medium text-sm transition-colors relative"
                    >
                        Invite Member
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-y-auto hide-scrollbar -mx-4 sm:-mx-6 px-4 sm:px-6">
                {loading ? (
                    renderSkeleton()
                ) : isMobileView ? (
                    filteredMembers.length > 0 ? (
                        <AutoSizer>
                            {({ height, width }) => (
                                <List
                                    className="hide-scrollbar" // NEW: Ensure List respects hide-scrollbar
                                    height={height}
                                    width={width}
                                    rowCount={filteredMembers.length}
                                    rowHeight={150}
                                    rowRenderer={renderMobileRow}
                                    overscanRowCount={5}
                                />
                            )}
                        </AutoSizer>
                    ) : (
                        <p className="text-center text-gray-500 dark:text-gray-400 py-8">No team members found matching your criteria.</p>
                    )
                ) : (
                    <div className="overflow-x-auto bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700">
                        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                            <thead className="bg-gray-50 dark:bg-gray-700/50">
                                <tr>
                                    {['Member', 'Role', 'Department', 'GPTs Assigned', 'Status', 'Joined', 'Last Active', ''].map((header) => (
                                        <th key={header} scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider whitespace-nowrap">
                                            {header}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                                {filteredMembers.length > 0 ? filteredMembers.map((member) => (
                                    <tr
                                        key={member.id}
                                        className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                                        onClick={() => user?._id !== member.id && handleViewMemberDetails(member)}
                                    >
                                        <td className="px-6 py-4 whitespace-nowrap cursor-pointer" onClick={() => user?._id !== member.id && handleViewMemberDetails(member)}>
                                            <div className="flex items-center">
                                                <div className="flex-shrink-0 h-10 w-10 rounded-full bg-gray-200 dark:bg-gray-600 flex items-center justify-center">
                                                    <FiUser className="text-gray-600 dark:text-gray-300" />
                                                </div>
                                                <div className="ml-4">
                                                    <div className="text-sm font-medium text-gray-900 dark:text-white">{member.name}</div>
                                                    <div className="text-sm text-gray-500 dark:text-gray-400">{member.email}</div>
                                                </div>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">{member.role}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700 dark:text-gray-300">{member.department}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-center text-gray-700 dark:text-gray-300">{member.assignedGPTs}</td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <span className={`px-2.5 py-0.5 inline-flex text-xs leading-5 font-semibold rounded-full ${member.status === 'Active'
                                                ? 'bg-green-100 dark:bg-green-900/50 text-green-800 dark:text-green-300'
                                                : 'bg-red-100 dark:bg-red-900/50 text-red-800 dark:text-red-300'}`}>
                                                {member.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{member.joined}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{member.lastActive}</td>
                                        <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium relative">
                                            <button
                                                onClick={(e) => { e.stopPropagation(); toggleActionsMenu(member.id); }}
                                                className={`text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-white p-1.5 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 ${user?._id === member.id ? 'opacity-50 cursor-not-allowed' : ''}`}
                                                data-member-id={member.id}
                                                disabled={user?._id === member.id}
                                                title={user?._id === member.id ? "You cannot modify your own account" : ""}
                                            >
                                                <FiMoreVertical size={18} />
                                            </button>
                                        </td>
                                    </tr>
                                )) : (
                                    <tr>
                                        <td colSpan="8" className="px-6 py-12 text-center text-sm text-gray-500 dark:text-gray-400">
                                            No team members found matching your criteria.
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {!loading && !error && filteredMembers.length > 0 && renderPagination()}

            {showActionsMenu && filteredMembers.map((member) => (
                member.id === showActionsMenu && user?._id !== member.id && (
                    <div
                        key={`menu-${member.id}`}
                        className="fixed z-50"
                        ref={node => {
                            if (node) {
                                const buttonRect = document.querySelector(`[data-member-id="${member.id}"]`)?.getBoundingClientRect();
                                if (buttonRect) {
                                    const menuHeight = 160; // Approximate height of the menu (adjust based on actual height)
                                    const viewportHeight = window.innerHeight;
                                    const isNearRightEdge = window.innerWidth - buttonRect.right < 200;
                                    const spaceBelow = viewportHeight - buttonRect.bottom;
                                    const showAbove = spaceBelow < menuHeight && buttonRect.top > menuHeight;

                                    // NEW: Position menu above if there's not enough space below
                                    if (showAbove) {
                                        node.style.top = `${buttonRect.top + window.scrollY - menuHeight - 5}px`;
                                    } else {
                                        node.style.top = `${buttonRect.bottom + window.scrollY + 5}px`;
                                    }

                                    if (isNearRightEdge) {
                                        node.style.right = `${window.innerWidth - buttonRect.right - window.scrollX}px`;
                                        node.style.left = 'auto';
                                    } else {
                                        node.style.left = `${buttonRect.left + window.scrollX - 60}px`;
                                        node.style.right = 'auto';
                                    }
                                }
                            }
                        }}
                    >
                        <div
                            ref={actionsMenuRef}
                            className="w-48 bg-white dark:bg-gray-800 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 dark:ring-gray-700 overflow-hidden"
                        >
                            <div className="py-1" role="menu" aria-orientation="vertical" aria-labelledby="options-menu">
                                <button onClick={() => handleAssignGpts(member)} className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2">
                                    <FiBox size={14} /> Assign GPTs
                                </button>
                                <button onClick={() => handleEditPermissions(member)} className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2">
                                    <FiEdit size={14} /> Edit Permissions
                                </button>
                                <button onClick={() => handleEmailTeamMember(member.email)} className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2">
                                    <FiMail size={14} /> Send Email
                                </button>
                                <div className="border-t border-gray-100 dark:border-gray-700 my-1"></div>
                                <button onClick={() => handleRemoveTeamMember(member.id)} className="w-full text-left px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center gap-2">
                                    <FiTrash2 size={14} /> Remove Member
                                </button>
                            </div>
                        </div>
                    </div>
                )
            ))}

            {showAssignGptsModal && selectedMemberForGpts && (
                <AssignGptsModal
                    isOpen={showAssignGptsModal}
                    onClose={() => setShowAssignGptsModal(false)}
                    teamMember={selectedMemberForGpts}
                    onAssignmentChange={handleGptAssignmentChange}
                />
            )}
            {showDetailsModal && selectedMemberForDetails && (
                <TeamMemberDetailsModal
                    isOpen={showDetailsModal}
                    onClose={() => setShowDetailsModal(false)}
                    member={selectedMemberForDetails}
                />
            )}
            {showInviteModal && (
                <InviteTeamMemberModal
                    isOpen={showInviteModal}
                    onClose={() => setShowInviteModal(false)}
                    onInviteSent={() => {
                        setPendingInvitesCount(prev => prev + 1);
                        toast.success("Invitation sent successfully");
                    }}
                />
            )}
            {showEditPermissionsModal && selectedMemberForPermissions && (
                <EditPermissionsModal
                    isOpen={showEditPermissionsModal}
                    onClose={() => setShowEditPermissionsModal(false)}
                    member={selectedMemberForPermissions}
                    onPermissionsUpdated={handlePermissionsUpdated}
                />
            )}
        </div>
    );
};

export default TeamManagement;