// Fixed video stream handling in client.js

let localStream;
let socket;
let roomId = '';
let username = '';
let isScreenSharing = false;
let screenStream = null;
let peers = {}; // Store all peer connections
let pendingCandidates = {}; // Store pending ICE candidates

// Chat functionality
let chatVisible = false;
let unreadMessages = 0;
let chatListenerRegistered = false;
let participants = new Map(); // Store participants with their info

// ASL Detection
let aslDetector = null;
let aslDetectionActive = false;

// Layout management
let activeParticipants = 0;

// Function to update video grid layout based on participant count
function updateVideoLayout() {
    const videoGrid = document.querySelector('.video-grid');
    if (!videoGrid) {
        console.warn('⚠️ Layout: Video grid not found');
        return;
    }

    // Count active video elements
    const activeVideos = videoGrid.querySelectorAll('.video-wrapper');
    const participantCount = activeVideos.length;

    console.log('🎬 Layout: Updating layout for', participantCount, 'participants');
    console.log('🎬 Layout: Current video grid classes:', videoGrid.className);

    // Remove all layout classes
    videoGrid.classList.remove('single-participant', 'two-participants', 'multiple-participants');

    // Apply appropriate layout class
    if (participantCount === 1) {
        videoGrid.classList.add('single-participant');
        console.log('📺 Layout: Applied single-participant layout (full screen)');
    } else if (participantCount === 2) {
        videoGrid.classList.add('two-participants');
        console.log('📺 Layout: Applied two-participants layout (split screen)');
    } else if (participantCount >= 3) {
        videoGrid.classList.add('multiple-participants');
        console.log('📺 Layout: Applied multiple-participants layout (grid)');
    }

    console.log('🎬 Layout: New video grid classes:', videoGrid.className);
    activeParticipants = participantCount;
    
    // Force a reflow to ensure styles are applied
    videoGrid.offsetHeight;
}

const configuration = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
        { urls: 'stun:stun2.l.google.com:19302' },
        { urls: 'stun:stun3.l.google.com:19302' },
        { urls: 'stun:stun4.l.google.com:19302' }
    ]
};

// Initialize the connection
async function init() {
    // Check if user is authenticated
    const token = localStorage.getItem('token');
    username = localStorage.getItem('username');
    
    if (!token || !username) {
        window.location.href = '/auth';
        return;
    }

    // Update username display
    const usernameDisplay = document.getElementById('username-display');
    if (usernameDisplay) {
        usernameDisplay.textContent = username;
    }
    
    // Initialize socket connection
    socket = io();
    roomId = new URLSearchParams(window.location.search).get('room') || 
             Math.random().toString(36).substring(7);
    
    // Update URL with room ID if not already there
    if (!window.location.search.includes('room')) {
        window.history.pushState(null, null, `?room=${roomId}`);
    }
    
    const roomInfo = document.getElementById('room-info');
    if (roomInfo) {
        roomInfo.textContent = `Room: ${roomId}`;
    }
    
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ 
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }, 
            audio: true 
        });
        
        // Display local video
        const localVideoElement = document.getElementById('localVideo');
        if (localVideoElement) {
            localVideoElement.srcObject = localStream;
            localVideoElement.autoplay = true;
            localVideoElement.playsInline = true;
            localVideoElement.muted = true; // Mute local video to prevent audio feedback
            
            // Update layout when local video is ready
            localVideoElement.addEventListener('loadedmetadata', () => {
                console.log('📹 Local video loaded, updating layout');
                updateVideoLayout();
            });
        }
        
        // Join the room with user name and token
        socket.emit('join', { 
            roomId: roomId, 
            userName: username,
            token: token
        });

        // Handle authentication errors
        socket.on('error', (error) => {
            console.error('Socket error:', error);
            if (error.message === 'Authentication failed') {
                localStorage.removeItem('token');
                localStorage.removeItem('username');
                window.location.href = '/auth';
            }
        });

        // Socket event handlers for multi-user connections
        socket.on('existingUsers', (existingUsers) => {
            console.log('Existing users in room:', existingUsers);
            existingUsers.forEach(user => {
                createPeerConnection(user.id, true, user.name);
                addParticipant(user.id, user.name);
            });
        });

        socket.on('userJoined', (user) => {
            console.log('New user joined:', user);
            createPeerConnection(user.id, false, user.name);
            addParticipant(user.id, user.name);
        });

        socket.on('offer', async ({ senderId, sdp }) => {
            console.log('Received offer from:', senderId);
            await handleOffer(senderId, sdp);
        });

        socket.on('answer', ({ senderId, sdp }) => {
            console.log('Received answer from:', senderId);
            handleAnswer(senderId, sdp);
        });

        socket.on('ice-candidate', ({ senderId, candidate }) => {
            handleIceCandidate(senderId, candidate);
        });

        socket.on('userLeft', (userId) => {
            console.log('User left:', userId);
            handleUserDisconnected(userId);
            removeParticipant(userId);
            updateVideoLayout();
        });

        // Handle incoming chat messages (remove any existing listeners first)
        socket.off('chatMessage');
        socket.on('chatMessage', (data) => {
            console.log('📨 socket.on chatMessage received:', data);
            // Check if this message is from the current user
            const isSent = data.sender === username;
            addMessage(data.message, data.sender, isSent);
        });

        // Handle ASL detection broadcasts from other users
        socket.off('asl-detection');
        socket.on('asl-detection', (data) => {
            console.log('🤟 ASL detection received from:', data.sender, '- word:', data.word);
            
            // Don't handle our own ASL detections (already handled locally)
            if (data.sender !== username) {
                // Show notification for other users' ASL
                showASLNotification(`${data.sender} signed: ${data.word.toUpperCase()}`, 'info');
                
                // Add to chat with ASL styling
                addMessage(data.word.toUpperCase(), data.sender, false, true);
            }
        });

        // UI event handlers
        document.getElementById('toggleMic')?.addEventListener('click', toggleMic);
        document.getElementById('toggleVideo')?.addEventListener('click', toggleVideo);
        document.getElementById('toggleScreen')?.addEventListener('click', toggleScreen);
        document.getElementById('endCall')?.addEventListener('click', endCall);
        
        // Perform initial layout update
        updateVideoLayout();

        // Initialize chat after socket connection
        setupChatUI();
        
        // Initialize participants list
        updateParticipantsList();
        
        // Initialize chat functionality
        initChat();
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to toggle chat
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                toggleChat();
            }
            
            // Escape to close chat
            if (e.key === 'Escape' && chatVisible) {
                e.preventDefault();
                toggleChat();
            }
        });
    } catch (err) {
        console.error('Error accessing media devices:', err);
        alert('Cannot access camera or microphone. Please check permissions: ' + err.message);
    }
}

function setupChatUI() {
    const sidebar = document.getElementById('chatSidebar');
    const toggleBtn = document.getElementById('toggleChat');
    const closeBtn = document.getElementById('closeChat');
    const sendBtn = document.getElementById('sendMessage');
    const input = document.getElementById('messageInput');

    // Sidebar toggle - use the proper toggleChat function
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleChat);
    }
    
    if (closeBtn) {
        closeBtn.addEventListener('click', toggleChat);
    }

    // Send message functionality
    if (sendBtn && input) {
        sendBtn.addEventListener('click', () => {
            const message = input.value.trim();
            if (message && socket && username && roomId) {
                socket.emit('chatMessage', {
                    roomId: roomId,
                    message: message,
                    sender: username
                });
                // Don't add message immediately - let server broadcast handle it
                input.value = '';
                sendBtn.disabled = true;
            }
        });

        // Auto-resize textarea and enable/disable send button
        input.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
            sendBtn.disabled = !this.value.trim();
        });

        // Send on Enter key (Shift+Enter for new line)
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendBtn.click();
            }
        });
    }
}

// Helper function to add a remote video element - FIXED
function addRemoteVideo(userId, stream, userName) {
    // Check if video element already exists
    let videoElement = document.querySelector(`video[data-peer="${userId}"]`);
    
    if (!videoElement) {
        console.log(`Creating new video element for user ${userId}`);
        // Create new video container
        const videoGrid = document.querySelector('.video-grid');
        if (!videoGrid) {
            console.error('Video grid not found in DOM');
            return null;
        }
        
        const videoWrapper = document.createElement('div');
        videoWrapper.className = 'video-wrapper';
        videoWrapper.setAttribute('data-peer-wrapper', userId);
        
        videoElement = document.createElement('video');
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        videoElement.setAttribute('data-peer', userId);
        
        const nameTag = document.createElement('div');
        nameTag.className = 'participant-name';
        nameTag.textContent = userName || `User ${userId.substring(0, 5)}`;
        
        // Add network quality indicator
        const qualityIndicator = document.createElement('div');
        qualityIndicator.className = 'network-quality';
        qualityIndicator.textContent = 'Connecting...';
        
        videoWrapper.appendChild(videoElement);
        videoWrapper.appendChild(nameTag);
        videoWrapper.appendChild(qualityIndicator);
        videoGrid.appendChild(videoWrapper);
        
        // Update layout after adding a new video
        updateVideoLayout();
    }
    
    // Set the stream as the source for this video element
    if (stream) {
        console.log(`Setting stream for user ${userId}`);
        videoElement.srcObject = stream;
        
        // Make sure we handle play promise correctly
        videoElement.play()
            .then(() => console.log(`Remote video for user ${userId} playing successfully`))
            .catch(e => {
                console.error(`Remote video play error for ${userId}:`, e);
                // Try again with user interaction handler
                videoElement.addEventListener('click', () => {
                    videoElement.play()
                        .then(() => console.log(`Remote video for user ${userId} playing after click`))
                        .catch(err => console.error(`Still failed to play: ${err}`));
                });
                
                // Show play overlay
                const playOverlay = document.createElement('div');
                playOverlay.className = 'play-overlay';
                playOverlay.textContent = 'Click to Play';
                videoWrapper.appendChild(playOverlay);
            });
    } else {
        console.warn(`Stream for user ${userId} is null or undefined`);
    }
    
    return videoElement;
}

// Create a peer connection for a specific user - FIXED
async function createPeerConnection(targetUserId, createOffer = false, userName) {
    if (peers[targetUserId]) {
        console.log(`Peer connection to ${targetUserId} already exists`);
        return peers[targetUserId];
    }

    console.log('Creating peer connection for:', targetUserId);
    const peer = new RTCPeerConnection(configuration);
    peers[targetUserId] = peer;

    // Initialize pending candidates array if needed
    if (!pendingCandidates[targetUserId]) {
        pendingCandidates[targetUserId] = [];
    }

    // Add local tracks to the peer connection
    localStream.getTracks().forEach(track => {
        console.log(`Adding ${track.kind} track to peer connection for ${targetUserId}`);
        peer.addTrack(track, localStream);
    });

    // Handle remote tracks - IMPROVED
    peer.ontrack = (event) => {
        console.log('Received remote track from:', targetUserId, event.track.kind);
        
        // Properly handle streams
        if (event.streams && event.streams[0]) {
            const stream = event.streams[0];
            console.log(`Adding stream with ${stream.getTracks().length} tracks from user ${targetUserId}`);
            
            // Add a small delay to ensure all tracks are received
            setTimeout(() => {
                addRemoteVideo(targetUserId, stream, userName);
            }, 100);
        } else {
            console.warn(`Received track without stream from ${targetUserId}`);
            // Create a new MediaStream if needed
            const newStream = new MediaStream();
            newStream.addTrack(event.track);
            addRemoteVideo(targetUserId, newStream, userName);
        }
    };

    // Handle ICE candidates - IMPROVED
    peer.onicecandidate = (event) => {
        if (event.candidate) {
            console.log(`Sending ICE candidate to ${targetUserId}`);
            socket.emit('ice-candidate', {
                targetUserId: targetUserId,
                candidate: event.candidate
            });
        } else {
            console.log(`ICE candidate gathering completed for ${targetUserId}`);
        }
    };

    // Ice connection state change monitoring - IMPROVED
    peer.oniceconnectionstatechange = () => {
        console.log(`ICE connection state with ${targetUserId}: ${peer.iceConnectionState}`);
        
        // Update UI based on connection state
        const qualityIndicator = document.querySelector(`[data-peer-wrapper="${targetUserId}"] .network-quality`);
        if (qualityIndicator) {
            if (peer.iceConnectionState === 'connected' || peer.iceConnectionState === 'completed') {
                qualityIndicator.className = 'network-quality good';
                qualityIndicator.textContent = 'Good Connection';
            } else if (peer.iceConnectionState === 'checking') {
                qualityIndicator.className = 'network-quality checking';
                qualityIndicator.textContent = 'Connecting...';
            } else if (peer.iceConnectionState === 'disconnected') {
                qualityIndicator.className = 'network-quality poor';
                qualityIndicator.textContent = 'Connection Issues';
                
                // Try to restart ICE after a brief delay if still disconnected
                setTimeout(() => {
                    if (peer.iceConnectionState === 'disconnected') {
                        try {
                            peer.restartIce();
                        } catch (e) {
                            console.warn('Could not restart ICE:', e);
                        }
                    }
                }, 2000);
            } else if (peer.iceConnectionState === 'failed') {
                qualityIndicator.className = 'network-quality failed';
                qualityIndicator.textContent = 'Connection Failed';
                
                // Try to restart ICE
                try {
                    peer.restartIce();
                } catch (e) {
                    console.warn('Could not restart ICE after failure:', e);
                    // If restart fails, recreate the connection after a delay
                    setTimeout(() => {
                        handleUserDisconnected(targetUserId);
                        createPeerConnection(targetUserId, true, userName);
                    }, 3000);
                }
            }
        }
    };

    // Connection state monitoring
    peer.onconnectionstatechange = () => {
        console.log(`Connection state with ${targetUserId}: ${peer.connectionState}`);
        if (peer.connectionState === 'failed') {
            console.log(`Connection with ${targetUserId} failed, attempting to reconnect`);
            // Close and recreate the peer connection
            handleUserDisconnected(targetUserId);
            setTimeout(() => createPeerConnection(targetUserId, true, userName), 2000);
        }
    };

    // Create offer if we initiated the connection
    if (createOffer) {
        try {
            const offer = await peer.createOffer({
                offerToReceiveAudio: true,
                offerToReceiveVideo: true
            });
            await peer.setLocalDescription(offer);
            console.log(`Sending offer to ${targetUserId}`);
            socket.emit('offer', {
                targetUserId: targetUserId,
                sdp: peer.localDescription
            });
        } catch (err) {
            console.error('Error creating offer:', err);
        }
    }
    
    return peer;
}

// Update video layout based on number of participants - IMPROVED
function updateVideoLayout() {
    const videoWrappers = document.querySelectorAll('.video-wrapper');
    const count = videoWrappers.length;
    
    console.log(`Updating video layout for ${count} participants`);
    
    // Calculate optimal grid dimensions
    let columns, rows;
    if (count <= 1) {
        columns = 1;
        rows = 1;
    } else if (count === 2) {
        columns = 2;
        rows = 1;
    } else if (count === 3) {
        columns = 2;
        rows = 2;
    } else if (count === 4) {
        columns = 2;
        rows = 2;
    } else {
        // For more than 4 participants, calculate optimal grid
        columns = Math.ceil(Math.sqrt(count));
        rows = Math.ceil(count / columns);
        
        // Adjust for better aspect ratio
        if (columns > rows) {
            const temp = columns;
            columns = rows;
            rows = temp;
        }
    }
    
    // Apply the calculated dimensions
    videoWrappers.forEach((wrapper, index) => {
        // Calculate position in grid
        const row = Math.floor(index / columns);
        const col = index % columns;
        
        // Set dimensions
        wrapper.style.width = `${100 / columns}%`;
        wrapper.style.height = `${100 / rows}%`;
        
        // Position the wrapper
        wrapper.style.position = 'absolute';
        wrapper.style.left = `${col * (100 / columns)}%`;
        wrapper.style.top = `${row * (100 / rows)}%`;
        
        // Ensure video fills the wrapper
        const video = wrapper.querySelector('video');
        if (video) {
            video.style.width = '100%';
            video.style.height = '100%';
            video.style.objectFit = 'cover';
        }
    });
    
    // Update the video grid container
    const videoGrid = document.querySelector('.video-grid');
    if (videoGrid) {
        videoGrid.style.display = 'grid';
        videoGrid.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
        videoGrid.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
    }
}

// Handle received offer - IMPROVED
async function handleOffer(senderId, sdp) {
    try {
        // Get or create peer connection
        let peer = peers[senderId];
        if (!peer) {
            peer = await createPeerConnection(senderId, false);
        }
        
        // Set remote description (the offer)
        console.log(`Setting remote description for offer from ${senderId}`);
        await peer.setRemoteDescription(new RTCSessionDescription(sdp));
        
        // Apply any pending ICE candidates
        if (pendingCandidates[senderId] && pendingCandidates[senderId].length > 0) {
            console.log(`Applying ${pendingCandidates[senderId].length} pending ICE candidates for ${senderId}`);
            
            const candidates = pendingCandidates[senderId];
            pendingCandidates[senderId] = [];
            
            for (const candidate of candidates) {
                try {
                    await peer.addIceCandidate(new RTCIceCandidate(candidate));
                    console.log(`Applied pending ICE candidate for ${senderId}`);
                } catch (err) {
                    console.error(`Error applying pending ICE candidate for ${senderId}:`, err);
                }
            }
        }
        
        // Create and send answer
        console.log(`Creating answer for ${senderId}`);
        const answer = await peer.createAnswer();
        console.log(`Setting local description (answer) for ${senderId}`);
        await peer.setLocalDescription(answer);
        
        console.log(`Sending answer to ${senderId}`);
        socket.emit('answer', {
            targetUserId: senderId,
            sdp: peer.localDescription
        });
    } catch (err) {
        console.error('Error handling offer:', err);
    }
}

// Handle received answer - IMPROVED
async function handleAnswer(senderId, sdp) {
    const peer = peers[senderId];
    if (peer) {
        try {
            console.log(`Setting remote description (answer) from ${senderId}`);
            await peer.setRemoteDescription(new RTCSessionDescription(sdp));
            
            // Apply any pending ICE candidates
            if (pendingCandidates[senderId] && pendingCandidates[senderId].length > 0) {
                console.log(`Applying ${pendingCandidates[senderId].length} pending ICE candidates after setting answer from ${senderId}`);
                
                const candidates = pendingCandidates[senderId];
                pendingCandidates[senderId] = [];
                
                for (const candidate of candidates) {
                    try {
                        await peer.addIceCandidate(new RTCIceCandidate(candidate));
                        console.log(`Applied pending ICE candidate for ${senderId}`);
                    } catch (err) {
                        console.error(`Error applying pending ICE candidate for ${senderId}:`, err);
                    }
                }
            }
        } catch (err) {
            console.error('Error handling answer:', err);
        }
    } else {
        console.warn(`Received answer from ${senderId} but no peer connection exists`);
    }
}

// Handle received ICE candidate - IMPROVED
async function handleIceCandidate(senderId, candidate) {
    const peer = peers[senderId];
    
    // Store candidate if peer doesn't exist yet or remote description isn't set
    if (!peer || !peer.remoteDescription || !peer.remoteDescription.type) {
        console.log(`Queueing ICE candidate from ${senderId} - peer or remote description not ready`);
        if (!pendingCandidates[senderId]) {
            pendingCandidates[senderId] = [];
        }
        pendingCandidates[senderId].push(candidate);
        return;
    }
    
    try {
        console.log(`Adding ICE candidate from ${senderId}`);
        await peer.addIceCandidate(new RTCIceCandidate(candidate));
    } catch (err) {
        console.error(`Error adding ICE candidate from ${senderId}:`, err);
        // Store as pending in case it was timing related
        if (!pendingCandidates[senderId]) {
            pendingCandidates[senderId] = [];
        }
        pendingCandidates[senderId].push(candidate);
    }
}

// Handle user disconnection
function handleUserDisconnected(userId) {
    console.log('User disconnected:', userId);
    
    // Close the peer connection
    if (peers[userId]) {
        try {
            // Stop all tracks
            peers[userId].getSenders().forEach(sender => {
                if (sender.track) {
                    sender.track.stop();
                }
            });
            
            // Close the peer connection
            peers[userId].close();
            delete peers[userId];
        } catch (error) {
            console.error('Error closing peer connection:', error);
        }
    }
    
    // Remove the video element
    const videoWrapper = document.querySelector(`[data-peer-wrapper="${userId}"]`);
    if (videoWrapper) {
        // Stop any tracks in the video element
        const videoElement = videoWrapper.querySelector('video');
        if (videoElement && videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
            videoElement.srcObject = null;
        }
        
        // Remove the wrapper element
        videoWrapper.remove();
        
        // Update layout after removing a video
        updateVideoLayout();
    }
    
    // Clear any pending ICE candidates
    if (pendingCandidates[userId]) {
        delete pendingCandidates[userId];
    }
    
    // Update the video layout
    updateVideoLayout();
}

// UI Controls - IMPROVED
function toggleMic() {
    const audioTrack = localStream.getAudioTracks()[0];
    if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        document.getElementById('toggleMic').classList.toggle('active');
        console.log(`Microphone ${audioTrack.enabled ? 'enabled' : 'disabled'}`);
    }
}

function toggleVideo() {
    const videoTrack = localStream.getVideoTracks()[0];
    if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        document.getElementById('toggleVideo').classList.toggle('active');
        console.log(`Camera ${videoTrack.enabled ? 'enabled' : 'disabled'}`);
    }
}

async function toggleScreen() {
    try {
        if (!isScreenSharing) {
            // Start screen sharing
            console.log('Starting screen sharing');
            screenStream = await navigator.mediaDevices.getDisplayMedia({ 
                video: {
                    cursor: 'always',
                    displaySurface: 'monitor'
                }
            });
            
            const videoTrack = screenStream.getVideoTracks()[0];
            if (!videoTrack) {
                throw new Error('No video track in screen sharing stream');
            }
            
            // Replace video track in all peer connections
            Object.values(peers).forEach(peer => {
                const sender = peer.getSenders().find(s => s.track && s.track.kind === 'video');
                if (sender) {
                    console.log('Replacing video track with screen share track');
                    sender.replaceTrack(videoTrack);
                }
            });
            
            // Update local video display
            document.getElementById('localVideo').srcObject = screenStream;
            
            // Handle screen sharing stop
            videoTrack.onended = async () => {
                await stopScreenSharing();
            };
            
            document.getElementById('toggleScreen').classList.add('active');
            isScreenSharing = true;
            
        } else {
            await stopScreenSharing();
        }
    } catch (err) {
        console.error('Error during screen sharing:', err);
        alert(`Screen sharing failed: ${err.message}. Please try again.`);
    }
}

async function stopScreenSharing() {
    if (screenStream) {
        console.log('Stopping screen sharing');
        screenStream.getTracks().forEach(track => track.stop());
        screenStream = null;
        
        // Restore camera video track in all peer connections
        const videoTrack = localStream.getVideoTracks()[0];
        if (videoTrack) {
            Object.values(peers).forEach(peer => {
                const sender = peer.getSenders().find(s => s.track && s.track.kind === 'video');
                if (sender) {
                    console.log('Replacing screen share track with camera track');
                    sender.replaceTrack(videoTrack);
                }
            });
        }
        
        // Update local video display
        document.getElementById('localVideo').srcObject = localStream;
        document.getElementById('toggleScreen').classList.remove('active');
        isScreenSharing = false;
    }
}

function endCall() {
    // Close all peer connections
    Object.values(peers).forEach(peer => {
        try {
            peer.close();
        } catch (e) {
            console.warn('Error closing peer connection:', e);
        }
    });
    peers = {};
    pendingCandidates = {};
    
    // Stop all media tracks
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
    }
    if (screenStream) {
        screenStream.getTracks().forEach(track => track.stop());
    }
    
    // Disconnect socket
    if (socket) {
        socket.disconnect();
    }
    
    // Redirect to home page
    window.location.href = 'premeeting.html';
}

// Handle page unload
window.addEventListener('beforeunload', function() {
    endCall();
});

// Initialize when the page loads
window.addEventListener('load', function() {
    init();
    
    // Copy room link functionality
    document.getElementById('copyLink')?.addEventListener('click', function() {
        const roomLink = window.location.href;
        navigator.clipboard.writeText(roomLink).then(() => {
            alert('Room link copied to clipboard!');
        }).catch(err => {
            console.error('Failed to copy room link:', err);
            // Fallback
            const linkElement = document.createElement('textarea');
            linkElement.value = roomLink;
            document.body.appendChild(linkElement);
            linkElement.select();
            document.execCommand('copy');
            document.body.removeChild(linkElement);
            alert('Room link copied to clipboard!');
        });
    });
    
    // Add these CSS styles for video grid
    const style = document.createElement('style');
    style.textContent = `
        .video-grid {
            display: grid;
            height: 100vh;
            width: 100vw;
            position: fixed;
            top: 0;
            left: 0;
            background: #1a1a1a;
            overflow: hidden;
            gap: 2px;
            padding: 2px;
        }
        .video-wrapper {
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            background: #000;
            border-radius: 4px;
        }
        .video-wrapper video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .participant-name {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 14px;
            z-index: 2;
            backdrop-filter: blur(5px);
        }
        .network-quality {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 2;
            backdrop-filter: blur(5px);
        }
        .network-quality.good {
            background: rgba(0, 255, 0, 0.7);
        }
        .network-quality.poor {
            background: rgba(255, 0, 0, 0.7);
        }
        .play-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 1;
            backdrop-filter: blur(5px);
        }
        @media (max-width: 768px) {
            .video-grid {
                gap: 1px;
                padding: 1px;
            }
            .participant-name {
                font-size: 12px;
                padding: 3px 6px;
            }
            .network-quality {
                font-size: 10px;
                padding: 3px 6px;
            }
        }
    `;
    document.head.appendChild(style);
    
    // Handle window resize for responsive layout
    window.addEventListener('resize', function() {
        updateVideoLayout();
    });
});

// Chat functionality
function toggleChat() {
    const chatSidebar = document.getElementById('chatSidebar');
    chatVisible = !chatVisible;
    if (chatVisible) {
        chatSidebar.classList.add('active');
        unreadMessages = 0;
        updateChatNotification();
        setTimeout(() => {
            document.getElementById('messageInput')?.focus();
        }, 300);
        console.log('Chat sidebar opened');
    } else {
        chatSidebar.classList.remove('active');
        console.log('Chat sidebar closed');
    }
}

function updateParticipantsList() {
    const participantsList = document.getElementById('participantsList');
    const participantCount = document.getElementById('participantCount');
    
    if (!participantsList || !participantCount) return;
    
    // Clear existing list
    participantsList.innerHTML = '';
    
    // Add current user first
    const currentUserItem = document.createElement('div');
    currentUserItem.className = 'participant-item';
    currentUserItem.innerHTML = `
        <div class="participant-avatar">
            <i class="fas fa-user"></i>
        </div>
        <div class="participant-info">
            <div class="participant-name">You (${username})</div>
            <div class="participant-status">Host</div>
        </div>
    `;
    participantsList.appendChild(currentUserItem);
    
    // Add other participants
    participants.forEach((participant, socketId) => {
        const participantItem = document.createElement('div');
        participantItem.className = 'participant-item';
        participantItem.innerHTML = `
            <div class="participant-avatar">
                <i class="fas fa-user"></i>
            </div>
            <div class="participant-info">
                <div class="participant-name">${participant.name}</div>
                <div class="participant-status">Participant</div>
            </div>
        `;
        participantsList.appendChild(participantItem);
    });
    
    // Update count
    participantCount.textContent = participants.size + 1; // +1 for current user
}

function addParticipant(socketId, name) {
    participants.set(socketId, { name });
    updateParticipantsList();
}

function removeParticipant(socketId) {
    participants.delete(socketId);
    updateParticipantsList();
}

function addMessage(message, sender, isSent = false, isASL = false) {
    const chatMessages = document.getElementById('chatMessages');
    
    // Remove empty state if it exists
    const emptyState = chatMessages.querySelector('.chat-empty');
    if (emptyState) {
        emptyState.remove();
    }
    
    const messageDiv = document.createElement('div');
    let messageClass = 'message';
    
    if (isASL) {
        messageClass += ' asl';
    } else {
        messageClass += isSent ? ' sent' : ' received';
    }
    
    messageDiv.className = messageClass;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    if (isASL) {
        messageDiv.innerHTML = `
            <div class="sender">ASL Detection</div>
            <div class="content">${escapeHtml(message)}</div>
            <div class="time">${time}</div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="sender">${sender}</div>
            <div class="content">${escapeHtml(message)}</div>
            <div class="time">${time}</div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Add notification if chat is not visible and message is not from current user
    if (!chatVisible && !isSent) {
        unreadMessages++;
        updateChatNotification();
        if (!isASL) { // Don't show chat notifications for ASL messages (they have their own notifications)
            showMessageNotification(sender, message);
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateChatNotification() {
    const notificationBadge = document.getElementById('chatNotification');
    if (notificationBadge) {
        if (unreadMessages > 0) {
            notificationBadge.textContent = unreadMessages > 99 ? '99+' : unreadMessages;
            notificationBadge.style.display = 'flex';
        } else {
            notificationBadge.style.display = 'none';
        }
    }
}

function showMessageNotification(sender, message) {
    // Show browser notification if supported
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(`New message from ${sender}`, {
            body: message.length > 50 ? message.substring(0, 50) + '...' : message,
            icon: '/favicon.ico'
        });
    }
}

// Initialize chat functionality
async function initChat() {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendMessage');
    
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
    
    // Fetch chat history
    try {
        const token = localStorage.getItem('token');
        const response = await fetch(`/api/chat/${roomId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const messages = await response.json();
            messages.forEach(msg => {
                addMessage(msg.message, msg.sender, msg.sender === username);
            });
        }
    } catch (error) {
        console.error('Error fetching chat history:', error);
    }
    
    // Add welcome message only if no messages exist
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages && !chatMessages.querySelector('.message')) {
        setTimeout(() => {
            addMessage('Welcome to the meeting! Feel free to use the chat to communicate.', 'System', false);
        }, 1000);
    }
}

// Function to handle new meeting creation
function createNewMeeting() {
    const roomId = Math.random().toString(36).substring(7);
    window.location.href = `/?room=${roomId}`;
}

// Function to handle joining a meeting
function joinMeeting() {
    const meetingCode = prompt('Enter meeting code:');
    if (meetingCode) {
        window.location.href = `/?room=${meetingCode}`;
    }
}

// Add event listeners for meeting buttons
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the app
    init();
    
    // Add meeting button listeners if on home page
    if (!window.location.search.includes('room')) {
        const newMeetingBtn = document.getElementById('newMeetingBtn');
        const joinMeetingBtn = document.getElementById('joinMeetingBtn');
        
        if (newMeetingBtn) {
            newMeetingBtn.addEventListener('click', createNewMeeting);
        }
        
        if (joinMeetingBtn) {
            joinMeetingBtn.addEventListener('click', joinMeeting);
        }
    }
    
    // Initialize ASL detection if on meeting page
    if (window.location.search.includes('room')) {
        // Wait for video to be ready before initializing ASL detection
        waitForVideoAndInitASL();
        
        // Initialize video layout
        setTimeout(() => {
            updateVideoLayout();
        }, 1000);
    }
});

// Handle window resize to update video layout
window.addEventListener('resize', () => {
    updateVideoLayout();
});

// Debug function to manually test layout (call from browser console)
window.testLayout = function(participantCount) {
    console.log('🧪 Testing layout with', participantCount, 'participants');
    const videoGrid = document.querySelector('.video-grid');
    
    // Remove all layout classes
    videoGrid.classList.remove('single-participant', 'two-participants', 'multiple-participants');
    
    if (participantCount === 1) {
        videoGrid.classList.add('single-participant');
    } else if (participantCount === 2) {
        videoGrid.classList.add('two-participants');
    } else {
        videoGrid.classList.add('multiple-participants');
    }
    
    console.log('🧪 Applied classes:', videoGrid.className);
};

// Function to wait for video element and ensure it's ready
function waitForVideoAndInitASL() {
    console.log('Waiting for video and dependencies...');
    
    const checkDependencies = () => {
        console.log('Checking dependencies:', {
            ASLDetector: typeof ASLDetector,
            CONFIG: typeof CONFIG,
            localVideo: !!document.getElementById('localVideo')
        });
        
        const localVideo = document.getElementById('localVideo');
        
        // Check if all dependencies are loaded
        if (typeof ASLDetector === 'undefined') {
            console.warn('ASLDetector not yet loaded, retrying...');
            setTimeout(checkDependencies, 500);
            return;
        }
        
        if (!localVideo) {
            console.warn('Local video element not found, retrying...');
            setTimeout(checkDependencies, 500);
            return;
        }
        
        if (localVideo.readyState >= 2) {
            // Video is ready, initialize ASL detection
            console.log('All dependencies ready, initializing ASL detection...');
            setTimeout(() => {
                initASLDetection();
            }, 1000);
        } else {
            // Video not ready yet, check again
            console.log('Video not ready yet, readyState:', localVideo.readyState);
            setTimeout(checkDependencies, 500);
        }
    };
    
    checkDependencies();
}

// ASL Detection Functions
function initASLDetection() {
    try {
        // Check if ASLDetector class is available
        if (typeof ASLDetector === 'undefined') {
            console.error('ASLDetector class not found. Make sure aslDetector.js is loaded.');
            showASLNotification('ASL detection not available. Please refresh the page.', 'error');
            return;
        }
        
        // Check if CONFIG is available
        if (typeof CONFIG === 'undefined') {
            console.warn('CONFIG not found, using default configuration');
        }
        
        // Initialize ASL detector
        try {
            aslDetector = new ASLDetector();
            // Expose on window for config modal updates
            window.aslDetector = aslDetector;
            console.log('ASL detector created successfully');
            
            // Wake up the API in the background
            showASLNotification('Initializing ASL detection...', 'info');
            aslDetector.wakeUpAPI().then(success => {
                if (success) {
                    showASLNotification('ASL detection ready', 'success');
                } else {
                    showASLNotification('ASL API may be slow to respond on first use', 'warning');
                }
            });
            
        } catch (e) {
            console.error('ASLDetector constructor failed:', e);
            showASLNotification(`Failed to initialize ASL detection: ${e.message || e}`, 'error');
            return;
        }
        
        // Initialize with saved configuration if available
        if (typeof initializeASLDetector === 'function') {
            try {
                initializeASLDetector();
            } catch (e) {
                console.warn('initializeASLDetector function failed:', e);
            }
        }
        
        // Set up result callback
        aslDetector.onResult((result) => {
            handleASLResult(result);
        });
        
        // Set up error callback
        aslDetector.onError((error) => {
            console.error('ASL detection error:', error);
            showASLNotification('ASL detection error: ' + (error.message || error), 'error');
        });
        
        console.log('ASL detection initialized successfully');
        showASLNotification('ASL detection ready', 'success');
        
    } catch (error) {
        console.error('Failed to initialize ASL detection:', error);
        showASLNotification(`Failed to initialize ASL detection: ${error.message || error}`, 'error');
    }
}

function toggleASLDetection() {
    console.log('toggleASLDetection called, aslDetector:', aslDetector);
    
    if (!aslDetector) {
        console.error('ASL detector not initialized');
        showASLNotification('ASL detector not initialized. Please refresh the page and try again.', 'error');
        
        // Try to reinitialize
        setTimeout(() => {
            console.log('Attempting to reinitialize ASL detection...');
            initASLDetection();
        }, 1000);
        return;
    }
    
    const localVideo = document.getElementById('localVideo');
    if (!localVideo) {
        console.error('Local video element not found');
        showASLNotification('Local video not found. Please ensure your camera is enabled.', 'error');
        return;
    }
    
    console.log('Local video found, readyState:', localVideo.readyState, 'dimensions:', localVideo.videoWidth, 'x', localVideo.videoHeight);
    
    // Check if video is ready
    if (!localVideo.videoWidth || !localVideo.videoHeight) {
        showASLNotification('Video not ready yet. Please wait a moment and try again.', 'warning');
        return;
    }
    
    if (aslDetectionActive) {
        // Stop detection
        aslDetector.stopDetection();
        aslDetectionActive = false;
        updateASLButton(false);
        showASLNotification('ASL detection stopped', 'info');
    } else {
        // Start detection
        // Use configured interval if available
        const interval = (typeof CONFIG !== 'undefined' && CONFIG.VIDEO?.CAPTURE_INTERVAL) ? CONFIG.VIDEO.CAPTURE_INTERVAL : 3000;
        console.log('Starting ASL detection with interval:', interval);
        aslDetector.startDetection(localVideo, interval);
        aslDetectionActive = true;
        updateASLButton(true);
        showASLNotification(`ASL detection started (every ${interval/1000}s)`, 'info');
    }
}

function handleASLResult(result) {
    try {
        console.log('🤟 ASL Result received:', result);
        
        if (!result || typeof result !== 'object') {
            console.warn('ASL result empty or invalid');
            return;
        }

        // Handle your API response format: {"detected_word": "word"}
        const detectedWord = result.detected_word;
        
        if (detectedWord && detectedWord.trim() !== '') {
            console.log('✅ ASL word detected:', detectedWord);
            
            // Show popup notification
            showASLNotification(`ASL Detected: ${detectedWord.toUpperCase()}`, 'success');
            
            // Add to chat automatically with ASL styling
            addMessage(detectedWord.toUpperCase(), `${username}`, false, true);
            
            // Broadcast to other users via socket
            if (socket && roomId) {
                socket.emit('asl-detection', {
                    roomId: roomId,
                    word: detectedWord,
                    sender: username,
                    timestamp: new Date().toISOString()
                });
            }
            
            return;
        }

        // If no word detected (empty string)
        console.log('ℹ️ No ASL sign detected in this frame');
        // Don't show notification for empty results to avoid spam
        
    } catch (e) {
        console.error('❌ Error handling ASL result:', e);
        showASLNotification('Error processing ASL detection result', 'error');
    }
}

function updateASLButton(isActive) {
    const aslButton = document.getElementById('aslToggleBtn');
    if (aslButton) {
        if (isActive) {
            aslButton.textContent = '🤟 Stop ASL';
            aslButton.className = 'control-button btn-danger';
            aslButton.title = 'Stop ASL Detection';
        } else {
            aslButton.textContent = '🤟 Start ASL';
            aslButton.className = 'control-button btn-success';
            aslButton.title = 'Start ASL Detection';
        }
    }
}

function showASLNotification(message, type = 'info') {
    // Remove any existing notifications to avoid clutter
    const existingNotifications = document.querySelectorAll('.asl-notification');
    existingNotifications.forEach(notification => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => notification.remove(), 300);
    });

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `asl-notification ${type}`;
    
    notification.innerHTML = `
        <span>${message}</span>
        <button type="button" onclick="this.parentElement.remove()" 
                style="background: none; border: none; color: rgba(255,255,255,0.8); 
                       font-size: 18px; cursor: pointer; padding: 0; margin-left: auto;">
            ×
        </button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after timeout (longer for success messages)
    const timeout = type === 'success' ? 5000 : 3000;
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => notification.remove(), 300);
        }
    }, timeout);
}