const express = require('express');
const https = require('https');
const fs = require('fs');
const { Server } = require('socket.io');
const path = require('path');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const cors = require('cors');

// MongoDB connection with error handling
mongoose.connect('mongodb://localhost:27017/videochat', {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => console.log('MongoDB connected successfully'))
.catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
});

// User schema
const userSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    createdAt: { type: Date, default: Date.now }
});

// Active user schema for video chat
const activeUserSchema = new mongoose.Schema({
    socketId: String,
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    name: String,
    roomId: String,
    createdAt: { type: Date, default: Date.now }
});

// Chat message schema
const chatMessageSchema = new mongoose.Schema({
    roomId: { type: String, required: true },
    sender: { type: String, required: true },
    message: { type: String, required: true },
    timestamp: { type: Date, default: Date.now }
});

const User = mongoose.model('User', userSchema);
const ActiveUser = mongoose.model('ActiveUser', activeUserSchema);
const ChatMessage = mongoose.model('ChatMessage', chatMessageSchema);

// Room management
const rooms = new Map();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// JWT Secret
const JWT_SECRET = 'your-secret-key'; // In production, use environment variable

// Authentication middleware
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    
    if (!token) {
        console.log('No token provided');
        return res.status(401).json({ message: 'Access denied' });
    }

    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        req.user = decoded;
        next();
    } catch (error) {
        console.error('Token verification failed:', error);
        return res.status(403).json({ message: 'Invalid token' });
    }
};

// Auth routes
app.post('/api/auth/signup', async (req, res) => {
    try {
        const { name, email, password } = req.body;

        // Check if user already exists
        const existingUser = await User.findOne({ email });
        if (existingUser) {
            return res.status(400).json({ message: 'User already exists' });
        }

        // Hash password
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(password, salt);

        // Create new user
        const user = new User({
            name,
            email,
            password: hashedPassword
        });

        await user.save();

        // Create token
        const token = jwt.sign({ id: user._id }, JWT_SECRET);
        
        // Return token and user data
        res.json({
            token,
            name: user.name
        });
    } catch (error) {
        console.error('Signup error:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

app.post('/api/auth/login', async (req, res) => {
    try {
        const { email, password } = req.body;

        // Check if user exists
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(400).json({ message: 'User not found' });
        }

        // Check password
        const validPassword = await bcrypt.compare(password, user.password);
        if (!validPassword) {
            return res.status(400).json({ message: 'Invalid password' });
        }

        // Create token
        const token = jwt.sign({ id: user._id }, JWT_SECRET);
        
        // Return token and user data
        res.json({
            token,
            name: user.name
        });
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// Get user data route
app.get('/api/user', authenticateToken, async (req, res) => {
    try {
        const user = await User.findById(req.user.id);
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }
        res.json({ name: user.name });
    } catch (error) {
        console.error('Error fetching user:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// Get chat history for a room
app.get('/api/chat/:roomId', authenticateToken, async (req, res) => {
    try {
        const { roomId } = req.params;
        const messages = await ChatMessage.find({ roomId })
            .sort({ timestamp: 1 })
            .limit(100); // Limit to last 100 messages
        
        res.json(messages);
    } catch (error) {
        console.error('Error fetching chat history:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// Load SSL certificates
const options = {
    key: fs.readFileSync('ssl/key.pem'),
    cert: fs.readFileSync('ssl/cert.pem')
};

// Create HTTPS server
const server = https.createServer(options, app);
const io = new Server(server);

// Socket.io connection handling
io.on('connection', async (socket) => {
    console.log('User connected:', socket.id);

    // Handle room joining
    socket.on('join', async ({ roomId, userName, token }) => {
        console.log('User', socket.id, 'joining room:', roomId, 'with name:', userName);
        
        try {
            // Verify token and get user ID
            const decoded = jwt.verify(token, JWT_SECRET);
            const userId = decoded.id;
            
            // Join the socket.io room
            socket.join(roomId);
            
            // Create the room if it doesn't exist
            if (!rooms.has(roomId)) {
                rooms.set(roomId, new Set());
            }
            
            const room = rooms.get(roomId);
            
            // Create or update active user in MongoDB
            const activeUser = await ActiveUser.findOneAndUpdate(
                { socketId: socket.id },
                { 
                    socketId: socket.id,
                    userId: userId, // Now using the actual user ID from the token
                    name: userName,
                    roomId: roomId
                },
                { upsert: true, new: true }
            );
            
            // Get all active users in the room
            const roomUsers = await ActiveUser.find({ roomId });
            const existingUsers = roomUsers
                .filter(u => u.socketId !== socket.id)
                .map(u => ({ id: u.socketId, name: u.name }));
                
            if (existingUsers.length > 0) {
                console.log('Sending existing users to', socket.id, ':', existingUsers);
                socket.emit('existingUsers', existingUsers);
            }
            
            // Notify all participants in the room about the new user
            socket.to(roomId).emit('userJoined', { id: socket.id, name: userName });
            
            // Add user to the room
            room.add(socket.id);
            
            // Store room ID in the socket for disconnect handling
            socket.roomId = roomId;
            
            console.log(`Room ${roomId} now has ${room.size} participants`);
        } catch (error) {
            console.error('Error joining room:', error);
            socket.emit('error', { message: 'Authentication failed' });
        }
    });

    // Handle direct signaling between peers
    socket.on('offer', ({ targetUserId, sdp }) => {
        console.log('Relaying offer from', socket.id, 'to', targetUserId);
        io.to(targetUserId).emit('offer', { senderId: socket.id, sdp });
    });

    socket.on('answer', ({ targetUserId, sdp }) => {
        console.log('Relaying answer from', socket.id, 'to', targetUserId);
        io.to(targetUserId).emit('answer', { senderId: socket.id, sdp });
    });

    socket.on('ice-candidate', ({ targetUserId, candidate }) => {
        io.to(targetUserId).emit('ice-candidate', { senderId: socket.id, candidate });
    });

    // Handle chat messages
    socket.on('chatMessage', async (data) => {
        try {
            // Store the message in MongoDB
            const chatMessage = new ChatMessage({
                roomId: data.roomId,
                sender: data.sender,
                message: data.message,
                timestamp: new Date()
            });
            await chatMessage.save();
            
            // Broadcast the message to all users in the room
            io.to(data.roomId).emit('chatMessage', {
                message: data.message,
                sender: data.sender,
                timestamp: new Date()
            });
        } catch (error) {
            console.error('Error saving chat message:', error);
        }
    });

    // Handle disconnection
    socket.on('disconnect', async () => {
        console.log('User disconnected:', socket.id);
        
        if (socket.roomId) {
            const room = rooms.get(socket.roomId);
            
            if (room) {
                // Remove user from the room
                room.delete(socket.id);
                
                // Remove active user from MongoDB
                await ActiveUser.findOneAndDelete({ socketId: socket.id });
                
                // Notify all participants about the user leaving
                io.to(socket.roomId).emit('userLeft', socket.id);
                
                console.log(`Room ${socket.roomId} now has ${room.size} participants`);
                
                // Clean up empty rooms
                if (room.size === 0) {
                    console.log(`Removing empty room: ${socket.roomId}`);
                    rooms.delete(socket.roomId);
                }
            }
        }
    });
});

// Serve the main page with authentication check
app.get('/', (req, res) => {
    res.redirect('/auth');
});

// Serve the meeting page with authentication check
app.get('/meeting', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'meeting.html'));
});

// Serve the auth page
app.get('/auth', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'auth.html'));
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
    console.log(`Secure server running at https://0.0.0.0:${PORT}/`);
});