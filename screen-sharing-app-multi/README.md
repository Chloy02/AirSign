# Screen Sharing App with Chat

A real-time video conferencing application with screen sharing and chat functionality similar to Google Meet.

## Features

### Video Conferencing
- Real-time video and audio communication
- Screen sharing capability
- Multiple participants support
- WebRTC peer-to-peer connections

### Chat System
- **Real-time messaging**: Send and receive messages instantly
- **Message persistence**: Chat history is saved and loaded when joining a room
- **Participant list**: See who's currently in the meeting
- **Notifications**: Browser notifications for new messages when chat is closed
- **Unread message counter**: Badge showing number of unread messages
- **Auto-resize textarea**: Input field automatically resizes as you type
- **Keyboard shortcuts**: 
  - `Ctrl/Cmd + Enter`: Toggle chat
  - `Escape`: Close chat
  - `Enter`: Send message
  - `Shift + Enter`: New line

### UI Features
- **Modern design**: Clean, Google Meet-inspired interface
- **Responsive layout**: Works on desktop and mobile devices
- **Dark theme**: Easy on the eyes for long meetings
- **Smooth animations**: Chat sidebar slides in/out smoothly

## How to Use Chat

1. **Open Chat**: Click the chat icon in the top-right corner or press `Ctrl/Cmd + Enter`
2. **Send Message**: Type in the input field and press Enter or click the send button
3. **View Participants**: See the list of participants in the meeting
4. **Chat History**: Previous messages are automatically loaded when you join
5. **Notifications**: Allow browser notifications to get notified of new messages

## Technical Details

### Backend
- **Node.js** with Express server
- **Socket.IO** for real-time communication
- **MongoDB** for storing chat messages and user data
- **JWT** for authentication

### Frontend
- **Vanilla JavaScript** with WebRTC
- **Responsive CSS** with modern design
- **Font Awesome** icons
- **Browser notifications** API

### Database Schema
```javascript
// Chat Message Schema
{
  roomId: String,
  sender: String,
  message: String,
  timestamp: Date
}
```

## Installation

1. Install dependencies:
```bash
npm install
```

2. Set up MongoDB:
```bash
# Make sure MongoDB is running on localhost:27017
```

3. Generate SSL certificates (for HTTPS):
```bash
# Place your SSL certificates in the ssl/ folder
# key.pem and cert.pem
```

4. Start the server:
```bash
npm start
```

5. Access the application:
```
https://localhost:3000
```

## Chat API Endpoints

- `GET /api/chat/:roomId` - Get chat history for a room
- `POST /api/auth/signup` - User registration
- `POST /api/auth/login` - User login

## Socket Events

- `chatMessage` - Send/receive chat messages
- `join` - Join a meeting room
- `userJoined` - New user joined
- `userLeft` - User left the meeting

## Browser Support

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Security Features

- JWT-based authentication
- HTTPS encryption
- Input sanitization for chat messages
- Secure WebRTC connections

## Future Enhancements

- File sharing in chat
- Emoji reactions
- Message editing/deletion
- Chat search functionality
- Message threading
- Read receipts 