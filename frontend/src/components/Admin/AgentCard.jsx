import React from 'react';
import { FiUsers, FiMessageSquare, FiGlobe } from 'react-icons/fi';
import { TbRouter } from 'react-icons/tb';
import { RiOpenaiFill } from 'react-icons/ri';

const AgentCard = ({ agentId, agentImage, agentName, status, userCount, messageCount, modelType, hasWebSearch, hideActionIcons }) => {
    const modelIcons = {
        'router-engine': <TbRouter className="text-yellow-500" size={18} />,
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 shadow-sm hover:shadow-md transition-all">
            <div className="h-32 sm:h-36 relative overflow-hidden bg-gradient-to-br from-gray-100 to-gray-300 dark:from-gray-700 dark:to-gray-900">
                {agentImage ? (
                    <div className="w-full h-full overflow-hidden">
                        <img 
                            src={agentImage} 
                            alt={agentName} 
                            className="w-full h-full object-cover object-center"
                            style={{ objectPosition: 'center' }}
                            onError={(e) => {
                                e.target.onerror = null;
                                e.target.src = '/img.png'; // Fallback to default image
                            }}
                        />
                    </div>
                ) : (
                    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-blue-50/50 to-purple-100/50 dark:from-blue-900/30 dark:to-purple-900/30">
                        <span className="text-3xl sm:text-4xl text-gray-500/40 dark:text-white/30">{agentName.charAt(0)}</span>
                    </div>
                )}
            </div>
            
            <div className="p-4">
                <h3 className="font-medium text-base mb-1 truncate">{agentName}</h3>
                <div className="flex items-center text-xs text-gray-500 dark:text-gray-400 mb-2">
                    {modelIcons[modelType === 'openrouter/auto' ? 'router-engine' : modelType] && (
                        <span>{modelIcons[modelType === 'openrouter/auto' ? 'router-engine' : modelType]}</span>
                    )}
                    <span className="ml-1">{modelType === 'openrouter/auto' ? 'router-engine' : modelType}</span>
                </div>
                
                {hasWebSearch && (
                    <div className="flex items-center gap-1 text-xs text-blue-500 dark:text-blue-400 mb-2">
                        <FiGlobe size={12} />
                        <span>Web search</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AgentCard;