package messages

import (
	context "context"

	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

const _ = grpc.SupportPackageIsVersion7

const (
	Ponger_Ping_FullMethodName = "/messages.Ponger/Ping"
)

type PongerClient interface {
	Ping(ctx context.Context, in *PingMessage, opts ...grpc.CallOption) (*PongMessage, error)
}

type pongerClient struct {
	cc grpc.ClientConnInterface
}

func NewPongerClient(cc grpc.ClientConnInterface) PongerClient {
	return &pongerClient{cc}
}

func (c *pongerClient) Ping(ctx context.Context, in *PingMessage, opts ...grpc.CallOption) (*PongMessage, error) {
	out := new(PongMessage)
	err := c.cc.Invoke(ctx, Ponger_Ping_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

type PongerServer interface {
	Ping(context.Context, *PingMessage) (*PongMessage, error)
	mustEmbedUnimplementedPongerServer()
}

type UnimplementedPongerServer struct {
}

func (UnimplementedPongerServer) Ping(context.Context, *PingMessage) (*PongMessage, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Ping not implemented")
}
func (UnimplementedPongerServer) mustEmbedUnimplementedPongerServer() {}

type UnsafePongerServer interface {
	mustEmbedUnimplementedPongerServer()
}

func RegisterPongerServer(s grpc.ServiceRegistrar, srv PongerServer) {
	s.RegisterService(&Ponger_ServiceDesc, srv)
}

func _Ponger_Ping_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PingMessage)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(PongerServer).Ping(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Ponger_Ping_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(PongerServer).Ping(ctx, req.(*PingMessage))
	}
	return interceptor(ctx, in, info, handler)
}

var Ponger_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "messages.Ponger",
	HandlerType: (*PongerServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Ping",
			Handler:    _Ponger_Ping_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "protos.proto",
}
