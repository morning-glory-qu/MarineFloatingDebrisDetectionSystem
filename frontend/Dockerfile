FROM node:24.12

RUN npm install -g pnpm

ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

WORKDIR /app

COPY . .

RUN pnpm install

EXPOSE 5173

CMD ["pnpm", "dev"]